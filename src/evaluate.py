"""Evaluacion del VAE: umbral de anomalia, severidad, metricas y monto en riesgo.

Lee reports/reconstruction_errors.csv (salida de la parte 3) y lo convierte
en umbrales de severidad, metricas de clasificacion y monto en riesgo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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


if __name__ == "__main__":
    df = load_errors()
    print(f"Filas cargadas: {len(df)}")
    print(df.head(3))