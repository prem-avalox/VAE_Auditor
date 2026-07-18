# Martín Dávalos - Contexto del Restaurante y Datos Sinteticos

## Objetivo de esta entrega

Responsable: **Martín Dávalos**.

Esta entrega deja definido el problema de negocio, las reglas del restaurante, el esquema de datos, las etiquetas de anomalia y el dataset sintetico que usara el resto del equipo para entrenar y evaluar el VAE.

## Negocio seleccionado

**Tipo de MiPYME:** restaurante pequeno ecuatoriano con punto de venta.

**Nombre ficticio para el proyecto:** Restaurante Sabor Andino.

**Contexto operativo:**

- Atiende desayunos, almuerzos, platos fuertes, bebidas, postres y combos.
- Opera con cajeros, meseros, mesas fisicas y canales de retiro/domicilio.
- Maneja pagos en efectivo, tarjeta y transferencia.
- Aplica descuentos normales de 0%, 5%, 10%, 15% y ocasionalmente 20%.
- Registra ventas, devoluciones y anulaciones.

## Horarios asumidos

- Lunes a sabado: 08:00 a 22:00.
- Domingo: 08:00 a 16:00.

Los horarios con mayor volumen son desayuno, almuerzo y cena. El pico principal es el almuerzo.

## Variables del dataset

| Campo | Descripcion |
| --- | --- |
| `id_transaccion` | Identificador unico de la transaccion. |
| `fecha_hora` | Fecha y hora del registro. |
| `dia_semana` | Dia de la semana en formato numerico: lunes=0, domingo=6. |
| `hora_decimal` | Hora representada como numero decimal. |
| `turno` | Desayuno, almuerzo, cena o fuera_horario. |
| `cajero` | Cajero que registra la venta. |
| `mesero` | Mesero asociado a la orden. |
| `mesa_canal` | Mesa fisica, domicilio, retiro en local o app de delivery. |
| `categoria_producto` | Categoria principal del producto. |
| `producto` | Producto representativo de la transaccion. |
| `cantidad_items` | Cantidad aproximada de items del ticket. |
| `monto_bruto` | Valor antes de descuento. |
| `descuento_pct` | Porcentaje de descuento aplicado. |
| `monto_final` | Valor despues del descuento. |
| `metodo_pago` | Efectivo, tarjeta o transferencia. |
| `tipo_transaccion` | Venta, devolucion o anulacion. |
| `es_anomalia` | 0 si es normal, 1 si es anomala. |
| `tipo_anomalia` | Tipo de anomalia inyectada. |
| `descripcion_anomalia` | Explicacion corta de la anomalia. |
| `split_sugerido` | Uso recomendado: train_normal, validacion o prueba. |

## Anomalias documentadas

| Tipo | Regla sintetica | Justificacion de negocio |
| --- | --- | --- |
| `descuento_fuera_politica` | Descuentos entre 45% y 80%. | Un descuento muy alto puede indicar error o abuso de permisos. |
| `venta_fuera_horario` | Transacciones entre 23:00 y 04:59. | El restaurante deberia estar cerrado. |
| `devolucion_monto_alto` | Devoluciones entre USD 80 y USD 250. | Las devoluciones normales suelen ser de monto bajo. |
| `monto_extremo` | Tickets mucho mas altos que el promedio. | Puede indicar digitacion incorrecta o venta atipica que requiere revision. |
| `monto_inconsistente` | Montos cero o negativos. | Inconsistencia clara para una venta ordinaria. |
| `efectivo_descuento_alto` | Pago en efectivo con descuento de 35% a 60%. | Combinacion sensible para auditoria interna. |
| `empleado_anulaciones_altas` | Anulaciones asociadas a un mismo cajero/mesero. | Patron util para detectar revision por empleado. |

## Dataset generado

Archivo principal:

```text
data/ventas_restaurante_sinteticas.csv
```

Caracteristicas:

- 12.000 transacciones sinteticas.
- 85% normales y 15% anomalas.
- Fechas entre mayo y junio de 2026.
- Etiquetas listas para evaluacion.
- Columna `split_sugerido` para que el modelo entrene solo con transacciones normales.

## Recomendacion para el equipo de modelo

Para entrenar el VAE, usar unicamente las filas con:

```text
split_sugerido = train_normal
es_anomalia = 0
```

Para validacion y pruebas, usar las filas con:

```text
split_sugerido = validacion
split_sugerido = prueba
```

Esto permite evaluar si el error de reconstruccion del VAE es mas alto en las transacciones anomalas.

## Como regenerar el dataset

Desde la raiz del proyecto:

```bash
python3 src/generate_data.py
```

El script usa una semilla fija, por lo que genera el mismo dataset mientras no se cambie el codigo.
