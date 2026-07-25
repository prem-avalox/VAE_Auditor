"""
FastAPI Backend Service for VAE Auditor
Provides REST API endpoints for single transaction evaluation, batch processing,
metrics retrieval, and log inspection.
"""

import json
from typing import Optional, List, Dict, Any
from pathlib import Path
import pandas as pd
import io

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.audit_service import evaluate_raw_transaction, process_raw_batch, load_artifacts
from src.logger import log_info, log_error, LOG_FILE

app = FastAPI(
    title="Auditor de Ventas Inteligente — VAE API",
    description="API Backend en FastAPI para la evaluación de transacciones con Autoencoder Variacional (PyTorch).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TransactionInput(BaseModel):
    monto: float = Field(..., example=15.50, description="Monto final de la venta ($)")
    descuento: float = Field(0.0, example=0.0, description="Monto o porcentaje de descuento")
    hora: float = Field(12.0, example=14.5, description="Hora decimal (0-24)")
    dia_semana: Any = Field("Lunes", example="Viernes", description="Día de la semana o índice 0-6")
    metodo_pago: str = Field("Efectivo", example="Tarjeta crédito", description="Método de pago")
    num_items: int = Field(3, example=2, description="Número de ítems en la transacción")
    cajero: Optional[str] = Field("Cajero_1", example="Cajero_2")
    mesero: Optional[str] = Field("Mesero_1", example="Mesero_3")
    mesa: Optional[str] = Field("Mesa_1", example="Mesa_5")
    categoria_producto: Optional[str] = Field("Platos Principales", example="Bebidas")
    producto: Optional[str] = Field("Plato Ejecutivo", example="Jugo Natural")
    tipo_transaccion: Optional[str] = Field("Venta", example="Venta")


class BatchResponse(BaseModel):
    total_transacciones: int
    total_anomalias: int
    monto_en_riesgo: float
    tiempo_total_segundos: float
    latencia_promedio_segundos: float
    throughput_transacciones_por_segundo: float
    resumen_severidad: Dict[str, int]


@app.get("/")
def read_root():
    """Verifica el estado de salud de la API y carga de artefactos."""
    try:
        load_artifacts()
        return {
            "status": "online",
            "service": "Auditor de Ventas VAE",
            "model_status": "loaded",
            "framework": "PyTorch + Scikit-Learn",
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
        }


@app.get("/health")
def health_check():
    return read_root()


@app.post("/evaluate")
def evaluate_transaction_endpoint(tx: TransactionInput):
    """
    Evalúa una transacción individual con el modelo VAE de PyTorch real.
    """
    try:
        tx_dict = tx.model_dump()
        resultado, performance = evaluate_raw_transaction(tx_dict)
        return {
            "evaluacion": resultado,
            "rendimiento": {
                "latencia_segundos": performance.latency,
                "tiempo_total_segundos": performance.total_time,
            }
        }
    except Exception as e:
        log_error("API_EVALUATE_ERROR", str(e))
        raise HTTPException(status_code=500, detail=f"Error evaluando transacción: {str(e)}")


@app.post("/batch", response_model=BatchResponse)
async def process_batch_endpoint(file: UploadFile = File(...)):
    """
    Recibe un archivo CSV o Excel y realiza la evaluación por lotes con el modelo VAE PyTorch.
    """
    try:
        contents = await file.read()
        filename = file.filename.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Formato de archivo no soportado. Usa CSV o Excel (.xlsx).")

        processed_df, performance = process_raw_batch(df)

        anomalias = processed_df[processed_df["prediccion_anomalia"] == 1]
        monto_col = "monto_final" if "monto_final" in processed_df.columns else ("monto" if "monto" in processed_df.columns else None)
        monto_riesgo = float(anomalias[monto_col].sum()) if monto_col else 0.0

        distribucion = processed_df["severidad"].value_counts().to_dict()

        return BatchResponse(
            total_transacciones=len(processed_df),
            total_anomalias=len(anomalias),
            monto_en_riesgo=round(monto_riesgo, 2),
            tiempo_total_segundos=round(performance.total_time, 6),
            latencia_promedio_segundos=round(performance.latency, 6),
            throughput_transacciones_por_segundo=round(performance.throughput, 2),
            resumen_severidad=distribucion,
        )

    except HTTPException:
        raise
    except Exception as e:
        log_error("API_BATCH_ERROR", str(e))
        raise HTTPException(status_code=500, detail=f"Error procesando archivo batch: {str(e)}")


@app.get("/metrics")
def get_model_metrics():
    """Retorna las métricas registradas del modelo VAE."""
    metrics_path = Path("reports/metricas_evaluacion.json")
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Métricas no encontradas en reports/metricas_evaluacion.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/logs")
def get_recent_logs(lines: int = Query(50, ge=1, le=500)):
    """Retorna las últimas N líneas del archivo de log del sistema."""
    if not LOG_FILE.exists():
        return {"logs": ["El archivo de logs aún no se ha generado."]}
    
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        recent = all_lines[-lines:]
        return {"total_lines": len(all_lines), "showing_lines": len(recent), "logs": [l.strip() for l in recent]}
