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

## Parte 5 - Frontend (Streamlit)

Interfaz web con dos perfiles, en `app.py`. Requiere que las partes 3 y 4 ya
se hayan ejecutado (usa los artefactos de `models/` y `reports/`).

```bash
streamlit run app.py
```

Usuarios de demostracion (ver `USERS` en `app.py`):

| Usuario   | Contrasena  | Rol      | Vista                                      |
|-----------|-------------|----------|---------------------------------------------|
| `tecnico` | `admin123`  | Tecnico  | Metricas, auditoria por lotes, logs         |
| `rosita`  | `rosita123` | Negocio  | Carga de Excel y reporte para el restaurante|

Credenciales fijas en el codigo, solo para la demostracion academica; no
estan pensadas para produccion (ver seccion de riesgos mas abajo).

**Perfil Tecnico:**

- Metricas de Rendimiento: precision, recall, F1, matriz de confusion y
  distribucion de severidad del split de prueba.
- Auditoria por Lotes (CSV/Excel): carga un archivo o usa el dataset de
  evaluacion incluido, con filtros por split/severidad/monto y descarga de
  resultados filtrados en Excel (coloreado por severidad, con filtros
  automaticos y encabezado congelado).
- Verificar Venta: evalua una transaccion individual en tiempo real contra
  el modelo VAE cargado en memoria.
- Logs y Rendimiento del Sistema: logs de auditoria en vivo y estado del
  backend.

**Perfil Negocio (Rosita):**

- Carga un Excel de ventas con las columnas `id_transaccion`, `fecha_hora`,
  `cajero`, `mesa`, `monto`, `descuento_pct`, `metodo_pago`,
  `tipo_transaccion`.
- El sistema corre cada transaccion por el modelo VAE real (via
  `src/audit_service.py`, no una simulacion) y devuelve un resumen
  ejecutivo con el monto en posible riesgo.
- Descarga de reporte en Excel con 3 hojas (Reporte Completo, Solo Alertas,
  Resumen Ejecutivo), coloreado por severidad y con filtros automaticos.

## Parte 6 - Backend (FastAPI)

API REST en `src/api.py`, construida sobre `src/audit_service.py` (el mismo
modelo VAE y preprocesador que usa el frontend, sin logica duplicada).
Pensada para atender peticiones concurrentes de 3 o mas usuarios
simultaneos.

Para iniciar el servidor:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

En Windows, si el comando `uvicorn` no se reconoce en la terminal, usar:

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Con el servidor corriendo, la documentacion interactiva (Swagger) queda
disponible en `http://localhost:8000/docs`, siempre sincronizada con el
codigo real.

### Endpoints

| Metodo | Ruta        | Descripcion                                                |
|--------|-------------|-------------------------------------------------------------|
| GET    | `/health`   | Estado de la API y del modelo (alias de `/`)                |
| POST   | `/evaluate` | Evaluacion de una transaccion individual con el VAE (PyTorch) en vivo |
| POST   | `/batch`    | Evaluacion por lotes de un archivo CSV o Excel, con throughput y latencia |
| GET    | `/metrics`  | Precision, recall, F1, matriz de confusion y monto en riesgo (desde `reports/metricas_evaluacion.json`) |
| GET    | `/logs`     | Ultimas N lineas del log de auditoria (`?lines=`, por defecto 50, maximo 500) |

### Pruebas de rendimiento y concurrencia

- `test_concurrency.py`: simula 3 usuarios concurrentes llamando al modelo
  en paralelo (thread-safe) y reporta latencia promedio y throughput.

  ```bash
  python test_concurrency.py
  ```

- `locustfile.py`: prueba de carga contra el backend real levantado con
  Uvicorn, golpeando `/evaluate`, `/metrics`, `/health` y `/logs`.

  ```bash
  locust -f locustfile.py --host=http://localhost:8000
  ```

- `test_logger.py` / `test_metrics.py`: pruebas unitarias de
  `src/logger.py` y `src/metrics.py`.

Modulos de soporte:

- `src/logger.py`: registra eventos de auditoria en `logs/app.log`.
- `src/metrics.py`: mide latencia, tiempo total y throughput por
  transaccion o por lote.
- `src/inference.py`: clasifica un error de reconstruccion ya calculado
  (sin recalibrar umbrales).
- `src/audit_service.py`: pipeline completo desde una transaccion cruda
  hasta su severidad — carga `models/preprocessor.joblib` y
  `models/vae_model.pt`, calcula el error de reconstruccion real y clasifica
  con `src/inference.py`. Es el unico punto de entrada tanto para el
  frontend (Parte 5) como para la API (Parte 6), para no duplicar logica de
  inferencia entre los dos.

## Riesgos y limitaciones conocidas

Para tener presente al presentar el proyecto, en la seccion de riesgos y
sostenibilidad:

- Autenticacion con credenciales fijas en el codigo (`USERS` en `app.py`):
  suficiente para la demostracion, no para produccion.
- Sin multi-tenancy: los datos de distintos restaurantes no estan aislados
  entre si.
- Persistencia en archivos (CSV/JSON), sin base de datos.
- Sin cifrado de datos sensibles (montos, nombres de cajeros/meseros).
- Los umbrales de severidad se calibraron una sola vez con el dataset
  sintetico; en un despliegue real conviene recalibrar por cliente y de
  forma periodica.
  
