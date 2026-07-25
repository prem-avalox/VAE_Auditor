from __future__ import annotations

from src.evaluate import classify_transaction
from src.logger import log_info, log_error
from src.metrics import Metrics


def evaluar_transaccion(
    reconstruction_error: float,
    thresholds: dict,
    monto: float | None = None,
):
    """
    Evalúa UNA transacción, añadiendo medición de rendimiento y logging.
    """

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
            (
                f"severidad={resultado['severidad']}, "
                f"latencia={performance.latency:.6f}s"
            ),
        )

        return resultado, performance

    except Exception as e:

        log_error(
            "TRANSACTION_ERROR",
            str(e),
        )

        raise

    import pandas as pd


def evaluar_csv(
    df: pd.DataFrame,
    reconstruction_error_column: str,
    thresholds: dict,
    monto_column: str | None = None,
):
    """
    Evalúa todas las transacciones de un DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame con las transacciones.

    reconstruction_error_column : str
        Nombre de la columna que contiene el error de reconstrucción.

    thresholds : dict
        Umbrales calculados por evaluate.py

    monto_column : str | None
        Columna del monto (si existe).
    """

    metrics = Metrics()
    metrics.start()

    resultados = []

    for _, row in df.iterrows():

        monto = None

        if monto_column is not None and monto_column in row:
            monto = row[monto_column]

        resultado = classify_transaction(
            reconstruction_error=row[reconstruction_error_column],
            thresholds=thresholds,
            monto=monto,
        )

        resultados.append(resultado)

    performance = metrics.stop(len(df))

    log_info(
        "CSV_PROCESSED",
        (
            f"transacciones={len(df)}, "
            f"latencia={performance.latency:.6f}s, "
            f"throughput={performance.throughput:.2f} trans/s"
        ),
    )

    return resultados, performance