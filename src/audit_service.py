import json
import threading
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch

from src.evaluate import classify_transaction, assign_severity
from src.logger import log_info, log_error, log_warning
from src.metrics import Metrics
from src.train_vae import load_vae

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
MODEL_PATH = PROJECT_ROOT / "models" / "vae_model.pt"
CONFIG_PATH = PROJECT_ROOT / "models" / "vae_model_config.json"
THRESHOLDS_PATH = PROJECT_ROOT / "reports" / "umbral_severidad.json"

_lock = threading.Lock()
_preprocessor = None
_vae_model = None
_thresholds = None

COLUMNAS_NUMERICAS = ["cantidad_items", "monto_bruto", "descuento_pct", "monto_final"]
COLUMNAS_CICLICAS_GENERADAS = ["dia_sin", "dia_cos", "hora_sin", "hora_cos"]
COLUMNAS_CATEGORICAS = [
    "turno", "cajero", "mesero", "mesa_canal", "categoria_producto",
    "producto", "metodo_pago", "tipo_transaccion"
]
COLUMNAS_ENTRADA = COLUMNAS_NUMERICAS + COLUMNAS_CICLICAS_GENERADAS + COLUMNAS_CATEGORICAS

DIA_MAP = {
    "lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2,
    "jueves": 3, "viernes": 4, "sábado": 5, "sabado": 5, "domingo": 6
}

# ── Explicación basada en el modelo: qué variable confundió más al VAE ──────
# Solo se consideran campos que la persona realmente ingresa (monto, hora,
# descuento, cajero, mesa, método de pago, tipo de transacción). Se excluyen
# a propósito turno/mesero/categoria_producto/producto/cantidad_items: para
# la mayoría de transacciones (las que sube Rosita, o las de "Verificar
# Venta") esos campos casi nunca se ingresan y quedan con un valor por
# defecto — reportarlos como "la razón" sería atribuir la anomalía a un dato
# que la persona nunca dio.
GRUPOS_EXPLICABLES = [
    "monto_final", "descuento_pct",
    "dia_semana", "hora", "cajero", "mesa_canal",
    "metodo_pago", "tipo_transaccion",
]
GRUPO_LABEL_TECNICO = {
    "monto_final": "Monto de la transacción",
    "descuento_pct": "Descuento aplicado",
    "dia_semana": "Día de la semana",
    "hora": "Hora del día",
    "cajero": "Cajero",
    "mesa_canal": "Mesa / canal de venta",
    "metodo_pago": "Método de pago",
    "tipo_transaccion": "Tipo de transacción",
}
GRUPO_LABEL_NEGOCIO = {
    "monto_final": "el monto de la venta",
    "descuento_pct": "el descuento aplicado",
    "dia_semana": "el día en que se hizo",
    "hora": "la hora en que se hizo",
    "cajero": "quién la cobró",
    "mesa_canal": "la mesa o canal de venta",
    "metodo_pago": "el método de pago",
    "tipo_transaccion": "el tipo de transacción",
}


def _feature_group_for_column(col_name: str):
    """Traduce una columna de salida del preprocesador (ej.
    'categoricas__cajero_CAJ-001') a su variable original ('cajero').
    Devuelve None para columnas que no se usan en la explicación.
    'monto_bruto' se fusiona con 'monto_final': son el mismo concepto de
    negocio (monto de la venta) y reportarlos aparte duplicaba la frase."""
    resto = col_name.split("__", 1)[1] if "__" in col_name else col_name
    if resto in ("dia_sin", "dia_cos"):
        return "dia_semana"
    if resto in ("hora_sin", "hora_cos"):
        return "hora"
    if resto == "monto_bruto":
        return "monto_final"
    if resto in GRUPOS_EXPLICABLES:
        return resto
    for campo in sorted(GRUPOS_EXPLICABLES, key=len, reverse=True):
        if resto == campo or resto.startswith(campo + "_"):
            return campo
    return None


def _build_group_index(preprocessor):
    """Lista de grupos (o None) alineada 1 a 1 con las columnas de salida
    del preprocesador, para poder sumar el error de reconstrucción por
    variable original en vez de por columna codificada."""
    return [_feature_group_for_column(c) for c in preprocessor.get_feature_names_out()]


def _explicar_por_error(sq_err_row, group_index, estilo="tecnico", top_n=2):
    """
    A partir del error cuadrático por columna de UNA transacción, agrupa por
    variable original y arma una frase con la(s) variable(s) que más
    contribuyeron al error de reconstrucción del VAE. Devuelve None si el
    error está muy repartido entre muchas variables (sin un "culpable" claro).
    """
    aportes = {}
    for valor, grupo in zip(sq_err_row, group_index):
        if grupo is None:
            continue
        aportes[grupo] = aportes.get(grupo, 0.0) + float(valor)

    total = sum(aportes.values())
    if total <= 0:
        return None

    ranking = sorted(aportes.items(), key=lambda kv: kv[1], reverse=True)
    etiquetas = GRUPO_LABEL_TECNICO if estilo == "tecnico" else GRUPO_LABEL_NEGOCIO
    # Solo se reporta una variable si de verdad concentra una parte
    # significativa del error (>15%); si no, es más honesto decir que no
    # hay una causa puntual clara que inventar una.
    principales = [etiquetas[g] for g, v in ranking[:top_n] if v / total > 0.15]
    if not principales:
        return None
    if estilo == "tecnico":
        return " · ".join(principales)
    return "Lo más inusual fue " + " y ".join(principales)


def load_artifacts():
    """Carga thread-safe de los artefactos del modelo (preprocesador, VAE, umbrales)."""
    global _preprocessor, _vae_model, _thresholds
    if _preprocessor is not None and _vae_model is not None and _thresholds is not None:
        return _preprocessor, _vae_model, _thresholds

    with _lock:
        if _preprocessor is None:
            if not PREPROCESSOR_PATH.exists():
                raise FileNotFoundError(f"No se encontró {PREPROCESSOR_PATH}. Ejecuta preprocessing.py primero.")
            _preprocessor = joblib.load(PREPROCESSOR_PATH)

        if _vae_model is None:
            if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
                raise FileNotFoundError(f"No se encontró el modelo o configuración en {MODEL_PATH}.")
            _vae_model = load_vae(model_path=MODEL_PATH, config_path=CONFIG_PATH, device="cpu")

        if _thresholds is None:
            if not THRESHOLDS_PATH.exists():
                raise FileNotFoundError(f"No se encontraron los umbrales en {THRESHOLDS_PATH}. Ejecuta evaluate.py primero.")
            with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
                _thresholds = json.load(f)

    return _preprocessor, _vae_model, _thresholds


def _normalize_mesa_canal(raw):
    """
    Traduce el formato de mesa/canal que sube Rosita ('Mesa 1', 'Domicilio',
    'Retiro') al formato exacto con el que se entrenó el modelo ('mesa_01',
    'domicilio', 'retiro_local'). Sin esto, NINGUNA mesa se reconocía —
    'Mesa 1' y 'mesa_01' son categorías distintas para el codificador.
    """
    if raw is None:
        return "mesa_01"
    texto = str(raw).strip().lower()
    if "domicilio" in texto:
        return "domicilio"
    if "retiro" in texto:
        return "retiro_local"
    if "delivery" in texto or "app" in texto:
        return "app_delivery"
    digitos = "".join(c for c in texto if c.isdigit())
    if "mesa" in texto and digitos:
        return f"mesa_{int(digitos):02d}"
    return texto


def _prepare_transaction_row(tx_dict: dict) -> pd.DataFrame:
    """Prepara y normaliza los campos de una transacción individual en un DataFrame."""
    monto = float(tx_dict.get("monto", tx_dict.get("monto_final", 10.0)))
    descuento = float(tx_dict.get("descuento", tx_dict.get("descuento_pct", 0.0)))
    
    # Calcular monto bruto si se pasó descuento
    monto_bruto = float(tx_dict.get("monto_bruto", monto + descuento))
    descuento_pct = (descuento / max(monto_bruto, 0.01)) * 100.0 if "descuento" in tx_dict and "descuento_pct" not in tx_dict else float(tx_dict.get("descuento_pct", 0.0))
    
    # Día de la semana y hora: si no vienen explícitos pero sí viene
    # fecha_hora (el caso normal cuando se sube un Excel/CSV con esa sola
    # columna), se derivan de ahí. Antes, sin este fallback, TODA
    # transacción sin dia_semana/hora explícitos caía en el valor por
    # defecto (lunes, 12:00) sin importar su fecha real — lo que distorsionaba
    # el error de reconstrucción de lotes completos.
    fecha_hora_val = tx_dict.get("fecha_hora")
    dia_input = tx_dict.get("dia_semana")
    hora_raw = tx_dict.get("hora", tx_dict.get("hora_decimal"))

    if (dia_input is None or hora_raw is None) and fecha_hora_val is not None:
        try:
            ts = pd.to_datetime(fecha_hora_val)
            if dia_input is None:
                dia_input = int(ts.weekday())
            if hora_raw is None:
                hora_raw = ts.hour + ts.minute / 60.0
        except (ValueError, TypeError):
            pass

    if dia_input is None:
        dia_input = 0
    if isinstance(dia_input, str):
        dia_idx = DIA_MAP.get(dia_input.strip().lower(), 0)
    else:
        dia_idx = int(dia_input) % 7

    # Hora decimal
    hora_val = float(hora_raw) if hora_raw is not None else 12.0

    row = {
        "cantidad_items": float(tx_dict.get("cantidad_items", tx_dict.get("num_items", 3))),
        "monto_bruto": monto_bruto,
        "descuento_pct": descuento_pct,
        "monto_final": monto,
        "dia_sin": np.sin(2 * np.pi * dia_idx / 7.0),
        "dia_cos": np.cos(2 * np.pi * dia_idx / 7.0),
        "hora_sin": np.sin(2 * np.pi * hora_val / 24.0),
        "hora_cos": np.cos(2 * np.pi * hora_val / 24.0),
        # Nota: turno/mesero/categoria_producto/producto casi nunca vienen en
        # los archivos que sube Rosita (su formato solo pide fecha_hora,
        # cajero, mesa, monto, descuento_pct, metodo_pago, tipo_transaccion).
        # Los valores por defecto de abajo son los MÁS FRECUENTES en los
        # datos de entrenamiento (no valores inventados), para minimizar el
        # error de reconstrucción artificial que meten categorías que el
        # modelo nunca vio.
        "turno": str(tx_dict.get("turno", "almuerzo")),
        "cajero": str(tx_dict.get("cajero", "CAJ-003")),
        "mesero": str(tx_dict.get("mesero", "MES-004")),
        "mesa_canal": _normalize_mesa_canal(tx_dict.get("mesa_canal", tx_dict.get("mesa"))),
        "categoria_producto": str(tx_dict.get("categoria_producto", "almuerzos")),
        "producto": str(tx_dict.get("producto", "Almuerzo ejecutivo")),
        # metodo_pago / tipo_transaccion sí los manda Rosita, pero en
        # Mayúscula ("Efectivo", "Venta"); el modelo se entrenó con minúscula
        # ("efectivo", "venta") — sin este .lower() cada transacción cae en
        # una categoría "desconocida" para el codificador.
        "metodo_pago": str(tx_dict.get("metodo_pago", "efectivo")).lower(),
        "tipo_transaccion": str(tx_dict.get("tipo_transaccion", "venta")).lower(),
    }
    return pd.DataFrame([row])


def evaluate_raw_transaction(tx_dict: dict):
    """
    Evalúa UNA transacción cruda usando el preprocesador y modelo PyTorch real.
    """
    metrics = Metrics()
    metrics.start()

    try:
        preprocessor, vae_model, thresholds = load_artifacts()

        df_row = _prepare_transaction_row(tx_dict)
        X_numpy = preprocessor.transform(df_row[COLUMNAS_ENTRADA])
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32)

        with torch.no_grad():
            reconstruction = vae_model.reconstruct_deterministic(X_tensor)
            sq_err = (X_tensor - reconstruction) ** 2
            rec_error = float(sq_err.mean(dim=1).numpy()[0])

        monto = float(tx_dict.get("monto", tx_dict.get("monto_final", 0.0)))
        resultado = classify_transaction(
            reconstruction_error=rec_error,
            thresholds=thresholds,
            monto=monto,
        )

        if resultado["es_anomalia"]:
            group_index = _build_group_index(preprocessor)
            sq_err_row = sq_err.numpy()[0]
            resultado["motivo_tecnico"] = _explicar_por_error(sq_err_row, group_index, "tecnico")
            resultado["motivo_negocio"] = _explicar_por_error(sq_err_row, group_index, "negocio")
        else:
            resultado["motivo_tecnico"] = None
            resultado["motivo_negocio"] = None

        performance = metrics.stop(1)

        log_info(
            "REAL_TRANSACTION_EVALUATED",
            f"severidad={resultado['severidad']}, rec_error={rec_error:.6f}, latencia={performance.latency:.6f}s"
        )

        return resultado, performance

    except Exception as e:
        log_error("RAW_TRANSACTION_ERROR", str(e))
        raise


def process_raw_batch(df: pd.DataFrame):
    """
    Procesa un lote de transacciones en pandas con el modelo VAE de PyTorch real.
    """
    metrics = Metrics()
    metrics.start()

    try:
        preprocessor, vae_model, thresholds = load_artifacts()
        processed_df = df.copy()

        # Si ya trae reconstruction_error precalculado (ej. CSV de evaluación), clasificamos directamente
        if "reconstruction_error" in processed_df.columns:
            processed_df["severidad"] = processed_df["reconstruction_error"].apply(
                lambda e: assign_severity(e, thresholds)
            )
            processed_df["prediccion_anomalia"] = (processed_df["severidad"] != "normal").astype(int)
            # No hay descomposición de error por variable disponible aquí
            # (el error ya venía calculado, no se recalcula con el modelo).
            processed_df["motivo_tecnico"] = None
            processed_df["motivo_negocio"] = None
        else:
            # Requerimos preprocesamiento completo
            rows = []
            for _, r in processed_df.iterrows():
                rows.append(_prepare_transaction_row(r.to_dict()).iloc[0])
            df_prepared = pd.DataFrame(rows)
            
            X_numpy = preprocessor.transform(df_prepared[COLUMNAS_ENTRADA])
            X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
            
            with torch.no_grad():
                reconstruction = vae_model.reconstruct_deterministic(X_tensor)
                sq_err_matrix = ((X_tensor - reconstruction) ** 2).numpy()
                rec_errors = sq_err_matrix.mean(axis=1)
            
            processed_df["reconstruction_error"] = rec_errors
            processed_df["severidad"] = [assign_severity(e, thresholds) for e in rec_errors]
            processed_df["prediccion_anomalia"] = (processed_df["severidad"] != "normal").astype(int)

            group_index = _build_group_index(preprocessor)
            processed_df["motivo_tecnico"] = [
                _explicar_por_error(fila, group_index, "tecnico") if sev != "normal" else None
                for fila, sev in zip(sq_err_matrix, processed_df["severidad"])
            ]
            processed_df["motivo_negocio"] = [
                _explicar_por_error(fila, group_index, "negocio") if sev != "normal" else None
                for fila, sev in zip(sq_err_matrix, processed_df["severidad"])
            ]

        # Ordenar por mayor error si la columna existe
        processed_df = processed_df.sort_values(by="reconstruction_error", ascending=False).reset_index(drop=True)
        total = len(processed_df)

        performance = metrics.stop(total)

        log_info(
            "REAL_BATCH_PROCESSED",
            f"transacciones={total}, tiempo_total={performance.total_time:.6f}s, "
            f"latencia={performance.latency:.6f}s, throughput={performance.throughput:.2f} trans/s"
        )

        return processed_df, performance

    except Exception as e:
        log_error("RAW_BATCH_ERROR", str(e))
        raise


def evaluate_transaction(reconstruction_error, thresholds, monto=None):
    """Función de legado para clasificar directamente por error de reconstrucción."""
    metrics = Metrics()
    metrics.start()
    try:
        resultado = classify_transaction(
            reconstruction_error=reconstruction_error,
            thresholds=thresholds,
            monto=monto,
        )
        performance = metrics.stop(1)
        log_info(
            "TRANSACTION_EVALUATED",
            f"severidad={resultado['severidad']}, latencia={performance.latency:.6f}s"
        )
        return resultado, performance
    except Exception as e:
        log_error("TRANSACTION_ERROR", str(e))
        raise


def process_batch(df: pd.DataFrame):
    """Función de legado para procesar lotes ya evaluados."""
    return process_raw_batch(df)