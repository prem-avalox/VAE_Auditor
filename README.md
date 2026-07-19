# VAE Auditor - Auditor de Ventas para Restaurante

Proyecto academico de Inteligencia Artificial y Aprendizaje Automatico.

## Objetivo

Construir un auditor inteligente de ventas para un restaurante MiPYME ecuatoriano. La solucion usa un Autoencoder Variacional (VAE) para aprender el patron normal de transacciones y detectar ventas, descuentos, devoluciones o anulaciones sospechosas.

## Aporte de Martín Dávalos

Responsable de la base del problema y los datos sinteticos:

- Definicion del contexto de negocio del restaurante.
- Reglas operativas: horarios, cajeros, meseros, mesas/canales, categorias, metodos de pago y descuentos.
- Tipos de anomalia documentados.
- Script reproducible para generar el dataset.
- Dataset sintetico con etiquetas para entrenamiento, validacion y prueba.

Documentos:

- `docs/martin_davalos_contexto_datos.md`
- `reports/martin_davalos_resumen_dataset.md`

## Estructura inicial

```text
RNA_IA/
  data/
    ventas_restaurante_sinteticas.csv
    README.md
  docs/
    martin_davalos_contexto_datos.md
  output/pdf/
    Guia_Inicio_Auditor_Ventas_VAE.pdf
  reports/
    martin_davalos_resumen_dataset.md
  src/
    generate_data.py
  tools/
    generate_auditor_ventas_pdf.py
  requirements.txt
```

## Dataset

Archivo:

```text
data/ventas_restaurante_sinteticas.csv
```

Resumen:

- 12.000 transacciones.
- 10.200 transacciones normales.
- 1.800 transacciones anomalas.
- 20 columnas.
- 7 tipos de anomalia.
- Split sugerido: `train_normal`, `validacion`, `prueba`.

## Regenerar datos

Desde la raiz del repositorio:

```bash
python3 src/generate_data.py
```

El script usa semilla fija para que el dataset sea reproducible.

## Instalacion sugerida

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Parte 3 - Entrenar el VAE

El preprocesamiento debe ejecutarse primero para generar los tensores. Luego,
desde la raiz del repositorio:

```bash
python src/train_vae.py
```

El entrenamiento usa exclusivamente `data/processed/train_tensor.pt` y valida
contra el CSV que sus 7.042 filas correspondan a `train_normal` y que ninguna
sea anomala. Los valores predeterminados son 50 epocas, batch de 128, learning
rate de 0.001, espacio latente de 8 dimensiones y beta de 0.001.

Salidas:

- `models/vae_model.pt`: pesos entrenados.
- `models/vae_model_config.json`: configuracion necesaria para recargarlo.
- `reports/vae_training_history.csv`: perdida por epoca.
- `reports/reconstruction_errors.csv`: MSE determinista por transaccion de
  validacion y prueba.

La explicacion tecnica y el contrato con la parte de evaluacion estan en
`reports/parte3_modelo_vae.md`.

## Parte 4 - Evaluacion

Toma el error de reconstruccion generado en la parte 3 y lo convierte en
umbral de anomalia, severidad, metricas de clasificacion y monto en riesgo.
Debe ejecutarse despues de `src/train_vae.py`. Desde la raiz del repositorio:

```bash
python src/evaluate.py
```

Los umbrales se calibran exclusivamente con las transacciones normales del
split de validacion; el split de prueba se usa una unica vez para reportar
las metricas finales, evitando fuga de datos.

Salidas:

- `reports/evaluacion_transacciones.csv`: severidad y prediccion por
  transaccion (validacion + prueba).
- `reports/umbral_severidad.json`: los 3 umbrales de severidad y su
  procedencia.
- `reports/metricas_evaluacion.json`: precision, recall, F1 y matriz de
  confusion por split, y monto en riesgo.
- `reports/distribucion_error_reconstruccion.png`: histograma del error,
  normal vs. anomalo, con los umbrales marcados.

Resultados actuales (split de prueba): precision 0.9322, recall 0.9557,
F1 0.9438, monto total en riesgo $86,964.11.

`src/evaluate.py` tambien expone `classify_transaction()`, para que la parte
5 clasifique una transaccion nueva en tiempo real cargando los umbrales ya
calibrados desde `reports/umbral_severidad.json`, sin recalibrar nada.

La explicacion tecnica completa esta en `reports/parte4_evaluacion.md`.
