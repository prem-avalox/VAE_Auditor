"""Evaluacion del VAE: umbral de anomalia, severidad, metricas y monto en riesgo.

Lee reports/reconstruction_errors.csv (salida de la parte 3) y lo convierte
en umbrales de severidad, metricas de clasificacion y monto en riesgo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
ERRORS_PATH = REPORTS_DIR / "reconstruction_errors.csv"

REQUIRED_COLUMNS = {
    "id_transaccion",
    "split",
    "monto_final",
    "es_anomalia",
    "tipo_anomalia",
    "reconstruction_error",
}

DEFAULT_LOW_PERCENTILE = 95.0
DEFAULT_MEDIUM_PERCENTILE = 99.0
DEFAULT_HIGH_PERCENTILE = 99.9

def load_errors() -> pd.DataFrame:
    """Carga el CSV de errores de reconstruccion generado por la parte 3."""
    if not ERRORS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro {ERRORS_PATH}. Ejecuta primero src/train_vae.py (parte 3)."
        )
    df = pd.read_csv(ERRORS_PATH)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas esperadas en {ERRORS_PATH}: {missing}")
    return df


def calibrate_thresholds(df: pd.DataFrame, low_pct: float, medium_pct: float, high_pct: float) -> dict:
    """Calibra los 3 umbrales de severidad usando solo transacciones
    normales del split de validacion (nunca el split de prueba)."""
    val_normal = df[(df["split"] == "validacion") & (df["es_anomalia"] == 0)]
    if val_normal.empty:
        raise ValueError("No hay transacciones normales en el split de validacion.")

    errors = val_normal["reconstruction_error"].to_numpy()
    return {
        "umbral_baja": float(np.percentile(errors, low_pct)),
        "umbral_media": float(np.percentile(errors, medium_pct)),
        "umbral_alta": float(np.percentile(errors, high_pct)),
        "percentiles_usados": {"baja": low_pct, "media": medium_pct, "alta": high_pct},
        "n_transacciones_calibracion": int(len(val_normal)),
        "fuente_calibracion": "validacion (solo transacciones normales, es_anomalia == 0)",
    }

def assign_severity(error: float, thresholds: dict) -> str:
    """Clasifica un error de reconstruccion individual en severidad."""
    if error <= thresholds["umbral_baja"]:
        return "normal"
    if error <= thresholds["umbral_media"]:
        return "baja"
    if error <= thresholds["umbral_alta"]:
        return "media"
    return "alta"


def classify_transaction(reconstruction_error: float, thresholds: dict, monto: float | None = None) -> dict:
    """Punto de entrada para la parte 5: clasifica UNA transaccion nueva
    a partir de su error de reconstruccion (ya calculado por la parte 3)
    y los umbrales ya calibrados (ver reports/umbral_severidad.json)."""
    severidad = assign_severity(reconstruction_error, thresholds)
    resultado = {
        "reconstruction_error": reconstruction_error,
        "severidad": severidad,
        "es_anomalia": severidad != "normal",
    }
    if monto is not None:
        resultado["monto"] = monto
    return resultado


def evaluate_split(df: pd.DataFrame, split_name: str, thresholds: dict) -> tuple[dict, pd.DataFrame]:
    """Aplica los umbrales a todas las transacciones de un split y calcula
    precision, recall, F1, matriz de confusion y monto en riesgo."""
    subset = df[df["split"] == split_name].copy()
    subset["severidad"] = subset["reconstruction_error"].apply(lambda e: assign_severity(e, thresholds))
    subset["prediccion_anomalia"] = (subset["severidad"] != "normal").astype(int)

    y_true = subset["es_anomalia"].to_numpy()
    y_pred = subset["prediccion_anomalia"].to_numpy()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    monto_en_riesgo = float(subset.loc[subset["prediccion_anomalia"] == 1, "monto_final"].sum())
    monto_en_riesgo_confirmado = float(
        subset.loc[(subset["prediccion_anomalia"] == 1) & (subset["es_anomalia"] == 1), "monto_final"].sum()
    )
    monto_total_split = float(subset["monto_final"].sum())

    metrics = {
        "split": split_name,
        "n_transacciones": int(len(subset)),
        "n_predichas_anomalas": int(y_pred.sum()),
        "n_reales_anomalas": int(y_true.sum()),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "matriz_confusion": {
            "verdaderos_negativos": int(tn),
            "falsos_positivos": int(fp),
            "falsos_negativos": int(fn),
            "verdaderos_positivos": int(tp),
        },
        "monto_total_split": round(monto_total_split, 2),
        "monto_total_en_riesgo": round(monto_en_riesgo, 2),
        "monto_en_riesgo_confirmado": round(monto_en_riesgo_confirmado, 2),
        "pct_monto_en_riesgo": round(100 * monto_en_riesgo / monto_total_split, 2) if monto_total_split else 0.0,
        "distribucion_severidad": subset["severidad"].value_counts().to_dict(),
    }
    return metrics, subset


def compute_error_gap(df: pd.DataFrame, split_name: str) -> dict:
    """Compara el error promedio de transacciones normales vs. anomalas.
    Util como dato de impacto para la presentacion (no afecta el umbral)."""
    subset = df[df["split"] == split_name]
    error_normal = subset.loc[subset["es_anomalia"] == 0, "reconstruction_error"].mean()
    error_anomalo = subset.loc[subset["es_anomalia"] == 1, "reconstruction_error"].mean()

    return {
        "split": split_name,
        "error_promedio_normal": round(float(error_normal), 6),
        "error_promedio_anomalo": round(float(error_anomalo), 6),
        "veces_mas_alto": round(float(error_anomalo / error_normal), 2) if error_normal else None,
    }

if __name__ == "__main__":
    df = load_errors()
    print(f"Filas cargadas: {len(df)}")

    thresholds = calibrate_thresholds(
        df, DEFAULT_LOW_PERCENTILE, DEFAULT_MEDIUM_PERCENTILE, DEFAULT_HIGH_PERCENTILE
    )
    print("\nUmbrales calibrados:")
    print(thresholds)

    test_metrics, test_scored = evaluate_split(df, "prueba", thresholds)
    print("\nMetricas - split de prueba:")
    print(test_metrics)

    gap = compute_error_gap(df, "prueba")
    print("\nComparacion de error normal vs. anomalo:")
    print(gap)