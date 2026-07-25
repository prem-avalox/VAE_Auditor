import json
import threading
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn.functional as F

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


def _prepare_transaction_row(tx_dict: dict) -> pd.DataFrame:
    """Prepara y normaliza los campos de una transacción individual en un DataFrame."""
    monto = float(tx_dict.get("monto", tx_dict.get("monto_final", 10.0)))
    descuento = float(tx_dict.get("descuento", tx_dict.get("descuento_pct", 0.0)))
    
    # Calcular monto bruto si se pasó descuento
    monto_bruto = float(tx_dict.get("monto_bruto", monto + descuento))
    descuento_pct = (descuento / max(monto_bruto, 0.01)) * 100.0 if "descuento" in tx_dict and "descuento_pct" not in tx_dict else float(tx_dict.get("descuento_pct", 0.0))
    
    # Día de la semana
    dia_input = tx_dict.get("dia_semana", 0)
    if isinstance(dia_input, str):
        dia_idx = DIA_MAP.get(dia_input.strip().lower(), 0)
    else:
        dia_idx = int(dia_input) % 7

    # Hora decimal
    hora_val = float(tx_dict.get("hora", tx_dict.get("hora_decimal", 12.0)))

    row = {
        "cantidad_items": float(tx_dict.get("cantidad_items", tx_dict.get("num_items", 3))),
        "monto_bruto": monto_bruto,
        "descuento_pct": descuento_pct,
        "monto_final": monto,
        "dia_sin": np.sin(2 * np.pi * dia_idx / 7.0),
        "dia_cos": np.cos(2 * np.pi * dia_idx / 7.0),
        "hora_sin": np.sin(2 * np.pi * hora_val / 24.0),
        "hora_cos": np.cos(2 * np.pi * hora_val / 24.0),
        "turno": str(tx_dict.get("turno", "Almuerzo")),
        "cajero": str(tx_dict.get("cajero", "Cajero_1")),
        "mesero": str(tx_dict.get("mesero", "Mesero_1")),
        "mesa_canal": str(tx_dict.get("mesa_canal", tx_dict.get("mesa", "Mesa_1"))),
        "categoria_producto": str(tx_dict.get("categoria_producto", "Platos Principales")),
        "producto": str(tx_dict.get("producto", "Plato Ejecutivo")),
        "metodo_pago": str(tx_dict.get("metodo_pago", "Efectivo")),
        "tipo_transaccion": str(tx_dict.get("tipo_transaccion", "Venta")),
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
            rec_error = float(F.mse_loss(reconstruction, X_tensor, reduction="none").mean(dim=1).numpy()[0])

        monto = float(tx_dict.get("monto", tx_dict.get("monto_final", 0.0)))
        resultado = classify_transaction(
            reconstruction_error=rec_error,
            thresholds=thresholds,
            monto=monto,
        )

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
                rec_errors = F.mse_loss(reconstruction, X_tensor, reduction="none").mean(dim=1).numpy()
            
            processed_df["reconstruction_error"] = rec_errors
            processed_df["severidad"] = [assign_severity(e, thresholds) for e in rec_errors]
            processed_df["prediccion_anomalia"] = (processed_df["severidad"] != "normal").astype(int)

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