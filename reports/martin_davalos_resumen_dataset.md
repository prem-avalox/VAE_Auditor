# Resumen del Dataset - Martín Dávalos

Responsable: **Martín Dávalos**.

## Archivo generado

```text
data/ventas_restaurante_sinteticas.csv
```

## Tamano del dataset

| Metrica | Valor |
| --- | ---: |
| Filas totales | 12.000 |
| Columnas | 20 |
| Transacciones normales | 10.200 |
| Transacciones anomalas | 1.800 |
| Porcentaje de anomalias | 15% |

## Distribucion por tipo de anomalia

| Tipo de registro | Cantidad |
| --- | ---: |
| normal | 10.200 |
| descuento_fuera_politica | 315 |
| monto_extremo | 269 |
| venta_fuera_horario | 256 |
| monto_inconsistente | 249 |
| devolucion_monto_alto | 244 |
| efectivo_descuento_alto | 241 |
| empleado_anulaciones_altas | 226 |

## Distribucion por split sugerido

| Split | Total | Normales | Anomalas |
| --- | ---: | ---: | ---: |
| train_normal | 7.042 | 7.042 | 0 |
| validacion | 2.251 | 1.602 | 649 |
| prueba | 2.707 | 1.556 | 1.151 |

## Uso recomendado

- Entrenamiento del VAE: usar solo `train_normal`.
- Calibracion de umbral: usar `validacion`.
- Evaluacion final: usar `prueba`.
- Campo objetivo para medir rendimiento: `es_anomalia`.
- Campo explicativo para reportes: `tipo_anomalia`.

## Nota para el grupo

El dataset es sintetico, por lo que sirve para una demo academica y para validar el pipeline completo. En una implementacion real, el siguiente paso seria reemplazarlo por transacciones anonimizadas de un restaurante real.
