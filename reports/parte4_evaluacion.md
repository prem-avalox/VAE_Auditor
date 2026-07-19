# Parte 4 - Evaluacion

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
(1602 transacciones). El split de
prueba nunca se usa para calibrar, solo para reportar metricas finales una
unica vez.

| Severidad | Percentil | Umbral (MSE) |
| --- | ---: | ---: |
| Baja | P95.0 | 0.063473 |
| Media | P99.0 | 0.082602 |
| Alta | P99.9 | 0.106123 |

## Metricas - split de prueba (evaluacion final)

- Precision: 0.9322
- Recall: 0.9557
- F1-score: 0.9438

| | Predicho normal | Predicho anomalo |
| --- | ---: | ---: |
| **Real normal** | 1476 (VN) | 80 (FP) |
| **Real anomalo** | 51 (FN) | 1100 (VP) |


## Monto en riesgo (split de prueba)

- Monto total en riesgo: $86,964.11 (83.30% del split)
- Monto en riesgo confirmado: $85,861.54

## Comparacion de error normal vs. anomalo

Una transaccion anomala tiene, en promedio, un error de reconstruccion
10.32 veces mas alto que una transaccion normal
(0.515799 vs. 0.049995).

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
