"""Evaluacion del VAE: umbral de anomalia, severidad, metricas y monto en riesgo.

Lee reports/reconstruction_errors.csv (salida de la parte 3) y lo convierte
en umbrales de severidad, metricas de clasificacion y monto en riesgo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import argparse
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
ERRORS_PATH = REPORTS_DIR / "reconstruction_errors.csv"
PLOT_PATH = REPORTS_DIR / "distribucion_error_reconstruccion.png"
EVAL_CSV_PATH = REPORTS_DIR / "evaluacion_transacciones.csv"
THRESHOLDS_JSON_PATH = REPORTS_DIR / "umbral_severidad.json"
METRICS_JSON_PATH = REPORTS_DIR / "metricas_evaluacion.json"
REPORT_MD_PATH = REPORTS_DIR / "parte4_evaluacion.md"

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


def make_plot(df_scored: pd.DataFrame, thresholds: dict) -> None:
    """Genera un histograma del error de reconstruccion, normal vs anomalo,
    con los 3 umbrales marcados. Se salta si no hay matplotlib instalado."""
    if not HAS_MATPLOTLIB:
        print("matplotlib no esta instalado, se omite el grafico.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    normal = df_scored[df_scored["es_anomalia"] == 0]["reconstruction_error"]
    anomalo = df_scored[df_scored["es_anomalia"] == 1]["reconstruction_error"]

    bins = np.linspace(0, df_scored["reconstruction_error"].quantile(0.99), 60)
    ax.hist(normal, bins=bins, alpha=0.6, label="Normal (real)", color="#1f77b4")
    ax.hist(anomalo, bins=bins, alpha=0.6, label="Anomala (real)", color="#d62728")

    for nombre, valor, color in [
        ("Umbral baja (P95)", thresholds["umbral_baja"], "#f0ad4e"),
        ("Umbral media (P99)", thresholds["umbral_media"], "#d9534f"),
        ("Umbral alta (P99.9)", thresholds["umbral_alta"], "#8b0000"),
    ]:
        ax.axvline(valor, color=color, linestyle="--", linewidth=1.2, label=nombre)

    ax.set_xlabel("Error de reconstruccion (MSE)")
    ax.set_ylabel("Numero de transacciones")
    ax.set_title("Distribucion del error de reconstruccion (split de prueba)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Grafico guardado en: {PLOT_PATH}")


def build_markdown_report(thresholds: dict, val_metrics: dict, test_metrics: dict, gap: dict) -> str:
    """Arma el reporte reports/parte4_evaluacion.md con los resultados
    ya calculados, siguiendo el mismo estilo que parte3_modelo_vae.md."""

    def cm_table(m: dict) -> str:
        cm = m["matriz_confusion"]
        return (
            "| | Predicho normal | Predicho anomalo |\n"
            "| --- | ---: | ---: |\n"
            f"| **Real normal** | {cm['verdaderos_negativos']} (VN) | {cm['falsos_positivos']} (FP) |\n"
            f"| **Real anomalo** | {cm['falsos_negativos']} (FN) | {cm['verdaderos_positivos']} (VP) |\n"
        )

    return f"""# Parte 4 - Evaluacion

## Objetivo

Convertir el error de reconstruccion del VAE (parte 3) en un umbral de
anomalia accionable, con severidad graduada, metricas de clasificacion y una
estimacion del monto en riesgo detectado.

## Entrada

Este modulo lee exclusivamente `reports/reconstruction_errors.csv`, generado
por `src/train_vae.py` (parte 3). No recalcula el error de reconstruccion ni
vuelve a tocar el modelo VAE.

## Calibracion del umbral (sin fuga de datos)

Los 3 umbrales se calibran usando unicamente las transacciones normales
(es_anomalia == 0) del split de validacion
({thresholds['n_transacciones_calibracion']} transacciones). El split de
prueba nunca se usa para calibrar, solo para reportar metricas finales una
unica vez.

| Severidad | Percentil | Umbral (MSE) |
| --- | ---: | ---: |
| Baja | P{thresholds['percentiles_usados']['baja']} | {thresholds['umbral_baja']:.6f} |
| Media | P{thresholds['percentiles_usados']['media']} | {thresholds['umbral_media']:.6f} |
| Alta | P{thresholds['percentiles_usados']['alta']} | {thresholds['umbral_alta']:.6f} |

## Metricas - split de prueba (evaluacion final)

- Precision: {test_metrics['precision']:.4f}
- Recall: {test_metrics['recall']:.4f}
- F1-score: {test_metrics['f1_score']:.4f}

{cm_table(test_metrics)}

## Monto en riesgo (split de prueba)

- Monto total en riesgo: ${test_metrics['monto_total_en_riesgo']:,.2f} ({test_metrics['pct_monto_en_riesgo']:.2f}% del split)
- Monto en riesgo confirmado: ${test_metrics['monto_en_riesgo_confirmado']:,.2f}

## Comparacion de error normal vs. anomalo

Una transaccion anomala tiene, en promedio, un error de reconstruccion
{gap['veces_mas_alto']} veces mas alto que una transaccion normal
({gap['error_promedio_anomalo']:.6f} vs. {gap['error_promedio_normal']:.6f}).

## Artefactos generados

| Archivo | Contenido |
| --- | --- |
| `reports/evaluacion_transacciones.csv` | Severidad y prediccion por transaccion |
| `reports/umbral_severidad.json` | Los 3 umbrales y su procedencia |
| `reports/metricas_evaluacion.json` | Metricas por split |
| `reports/distribucion_error_reconstruccion.png` | Histograma del error con umbrales |

## Contrato con la parte 5

La parte 5 puede llamar directamente a `classify_transaction()` para evaluar
una transaccion nueva en tiempo real, cargando los umbrales desde
`reports/umbral_severidad.json`, o usar `reports/evaluacion_transacciones.csv`
como fuente de datos ya evaluados para poblar la tabla y el dashboard.

## Ejecucion

```bash
python src/evaluate.py
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluacion del VAE: umbral, severidad y metricas.")
    parser.add_argument("--low-percentile", type=float, default=DEFAULT_LOW_PERCENTILE)
    parser.add_argument("--medium-percentile", type=float, default=DEFAULT_MEDIUM_PERCENTILE)
    parser.add_argument("--high-percentile", type=float, default=DEFAULT_HIGH_PERCENTILE)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_errors()
    thresholds = calibrate_thresholds(df, args.low_percentile, args.medium_percentile, args.high_percentile)

    val_metrics, val_scored = evaluate_split(df, "validacion", thresholds)
    test_metrics, test_scored = evaluate_split(df, "prueba", thresholds)
    gap = compute_error_gap(df, "prueba")

    scored_all = pd.concat([val_scored, test_scored], ignore_index=True).sort_values("id_transaccion")
    scored_all.to_csv(EVAL_CSV_PATH, index=False)

    THRESHOLDS_JSON_PATH.write_text(json.dumps(thresholds, indent=2, ensure_ascii=False), encoding="utf-8")
    METRICS_JSON_PATH.write_text(
        json.dumps({"validacion": val_metrics, "prueba": test_metrics}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    make_plot(test_scored, thresholds)

    REPORT_MD_PATH.write_text(build_markdown_report(thresholds, val_metrics, test_metrics, gap), encoding="utf-8")

    print("\nUmbrales calibrados:")
    print(json.dumps(thresholds, indent=2, ensure_ascii=False))
    print("\nMetricas - split de prueba:")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    print("\nComparacion de error normal vs. anomalo:")
    print(json.dumps(gap, indent=2, ensure_ascii=False))
    print(f"\nArtefactos escritos en: {REPORTS_DIR}")

if __name__ == "__main__":
    main()