"""Evaluacion del VAE: umbral de anomalia, severidad, metricas y monto en riesgo.

Lee reports/reconstruction_errors.csv (salida de la parte 3) y lo convierte
en umbrales de severidad, metricas de clasificacion y monto en riesgo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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

if __name__ == "__main__":
    df = load_errors()
    print(f"Filas cargadas: {len(df)}")

    thresholds = calibrate_thresholds(
        df, DEFAULT_LOW_PERCENTILE, DEFAULT_MEDIUM_PERCENTILE, DEFAULT_HIGH_PERCENTILE
    )
    print("\nUmbrales calibrados:")
    print(thresholds)

    ejemplo = classify_transaction(reconstruction_error=0.09, thresholds=thresholds, monto=45.50)
    print("\nEjemplo de clasificacion (parte 5 usaria esta funcion asi):")
    print(ejemplo)