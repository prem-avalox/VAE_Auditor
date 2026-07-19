# Parte 3 - Modelo VAE

## Objetivo

El modelo aprende el patron de las transacciones normales del Restaurante
Sabor Andino y produce un error de reconstruccion por transaccion. Un error
alto indica que el patron fue dificil de reconstruir y puede ser anomalo.

## Entrada y control de data leakage

El entrenamiento lee exclusivamente:

```text
data/processed/train_tensor.pt
```

Este tensor contiene 7.042 transacciones del split `train_normal`. Antes de
entrenar, el script contrasta su cantidad con el CSV fuente y detiene la
ejecucion si encuentra alguna fila con `es_anomalia != 0`. Las etiquetas de
validacion y prueba nunca se entregan al optimizador.

Los tensores `val_tensor.pt` y `test_tensor.pt` se cargan solamente para
calcular errores despues de finalizar el entrenamiento:

- Validacion: calibracion del umbral por la parte 4.
- Prueba: evaluacion final, despues de fijar el umbral.

## Arquitectura

El vector de entrada tiene actualmente 68 caracteristicas preprocesadas.

```text
Encoder:  68 -> Linear(64) -> ReLU -> Linear(32) -> ReLU
Latente:  mu = Linear(32, 8), logvar = Linear(32, 8)
Decoder:  8 -> Linear(32) -> ReLU -> Linear(64) -> ReLU -> Linear(68)
```

Durante el entrenamiento se aplica el truco de reparametrizacion:

```text
sigma = exp(0.5 * logvar)
z = mu + epsilon * sigma, con epsilon ~ N(0, I)
```

Esto conserva el muestreo del espacio latente y permite propagar gradientes.

## Perdida VAE

Se minimiza la version negativa de la ELBO utilizada por el proyecto:

```text
loss = MSE(x_reconstruido, x) + beta * KL(q(z|x) || N(0, I))
beta = 0.001
```

El MSE mide la calidad de reconstruccion. La divergencia KL regulariza el
espacio latente hacia una normal estandar. El KL se suma sobre las ocho
dimensiones latentes y se promedia entre las observaciones del lote.

## Hiperparametros predeterminados

| Parametro | Valor |
| --- | ---: |
| Epocas | 50 |
| Batch size | 128 |
| Learning rate | 0.001 |
| Latent dim | 8 |
| Beta | 0.001 |
| Semilla | 42 |

Estos valores corresponden al alcance acordado. El modelo y el dataset son
pequenos; extender el entrenamiento por horas no garantiza una mejora y puede
provocar sobreajuste. La curva guardada debe revisarse antes de aumentar las
epocas.

## Ejecucion

Desde la raiz del repositorio:

```bash
python src/train_vae.py
```

Los parametros pueden modificarse de forma explicita, por ejemplo:

```bash
python src/train_vae.py --epochs 50 --batch-size 128 --learning-rate 0.001
```

## Artefactos generados

| Archivo | Contenido |
| --- | --- |
| `models/vae_model.pt` | `state_dict` entrenado |
| `models/vae_model_config.json` | Arquitectura, dimensiones e hiperparametros |
| `reports/vae_training_history.csv` | Loss, MSE y KL por epoca |
| `reports/reconstruction_errors.csv` | MSE determinista por transaccion |

El reporte de errores incluye `id_transaccion`, `indice_split`, `split`,
`split_sugerido`, `es_anomalia`, `tipo_anomalia`, `monto_final` y
`reconstruction_error`. Las
columnas de etiqueta y monto son solo para la evaluacion posterior; no forman
parte del entrenamiento.

## Scoring reproducible

El entrenamiento muestrea `z`, como exige un VAE. En inferencia se reconstruye
con `decode(mu)` para evitar que una misma transaccion reciba un score diferente
en cada ejecucion. El score es:

```text
reconstruction_error = mean((x - reconstruction) ** 2, por feature)
```

La parte 4 debe elegir el umbral exclusivamente con las filas de validacion y
usar prueba una sola vez para precision, recall, F1 y matriz de confusion.

## Carga desde otro modulo

```python
from src.train_vae import load_vae

model = load_vae(device="cpu")
```

La funcion lee la configuracion, reconstruye la arquitectura, carga el
`state_dict` de forma estricta y devuelve el modelo en modo evaluacion.
