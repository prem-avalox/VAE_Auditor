import csv
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "ventas_restaurante_sinteticas.csv"
SEED = 20260716


@dataclass(frozen=True)
class MenuItem:
    categoria: str
    producto: str
    precio: float
    peso: int


MENU = [
    MenuItem("desayunos", "Bolon mixto", 3.50, 10),
    MenuItem("desayunos", "Tigrillo con cafe", 4.25, 8),
    MenuItem("desayunos", "Humita con cafe", 3.25, 6),
    MenuItem("almuerzos", "Almuerzo ejecutivo", 4.00, 22),
    MenuItem("almuerzos", "Seco de pollo", 5.50, 16),
    MenuItem("almuerzos", "Menestra con carne", 6.25, 14),
    MenuItem("almuerzos", "Encebollado", 4.75, 12),
    MenuItem("platos_fuertes", "Churrasco ecuatoriano", 8.75, 8),
    MenuItem("platos_fuertes", "Parrillada personal", 10.50, 6),
    MenuItem("platos_fuertes", "Camarones apanados", 11.25, 5),
    MenuItem("bebidas", "Jugo natural", 1.75, 14),
    MenuItem("bebidas", "Gaseosa personal", 1.25, 18),
    MenuItem("bebidas", "Agua sin gas", 1.00, 10),
    MenuItem("postres", "Tres leches", 2.75, 5),
    MenuItem("postres", "Flan de la casa", 2.50, 4),
    MenuItem("combos", "Combo familiar almuerzo", 18.00, 5),
    MenuItem("combos", "Combo para dos", 12.50, 7),
]

CAJEROS = ["CAJ-001", "CAJ-002", "CAJ-003", "CAJ-004"]
MESEROS = ["MES-001", "MES-002", "MES-003", "MES-004", "MES-005", "MES-006"]
METODOS_PAGO = ["efectivo", "tarjeta", "transferencia"]
TIPOS_TRANSACCION = ["venta", "devolucion", "anulacion"]
CANAL_MESA = [f"mesa_{i:02d}" for i in range(1, 16)] + ["domicilio", "retiro_local", "app_delivery"]


def weighted_choice(items, weight_attr=None):
    if weight_attr is None:
        return random.choice(items)
    total = sum(getattr(item, weight_attr) for item in items)
    pick = random.uniform(0, total)
    current = 0
    for item in items:
        current += getattr(item, weight_attr)
        if pick <= current:
            return item
    return items[-1]


def business_hours_for(date_obj):
    # Lunes=0 ... domingo=6. El restaurante cierra temprano los domingos.
    if date_obj.weekday() == 6:
        return 8, 16
    return 8, 22


def normal_datetime(start_date, end_date):
    days = (end_date - start_date).days
    date_obj = start_date + timedelta(days=random.randint(0, days))
    open_hour, close_hour = business_hours_for(date_obj)

    # Distribucion con picos en desayuno, almuerzo y cena.
    peaks = [9.0, 13.0, 19.5]
    peak_weights = [0.20, 0.50, 0.30]
    peak = random.choices(peaks, weights=peak_weights, k=1)[0]
    hour = random.gauss(peak, 1.15)
    hour = max(open_hour + 0.1, min(close_hour - 0.1, hour))
    minute = int((hour % 1) * 60)
    return datetime(date_obj.year, date_obj.month, date_obj.day, int(hour), minute)


def outside_hours_datetime(start_date, end_date):
    days = (end_date - start_date).days
    date_obj = start_date + timedelta(days=random.randint(0, days))
    hour = random.choice([0, 1, 2, 3, 4, 23])
    minute = random.randint(0, 59)
    return datetime(date_obj.year, date_obj.month, date_obj.day, hour, minute)


def turn_for_hour(hour):
    if hour < 11:
        return "desayuno"
    if hour < 16:
        return "almuerzo"
    return "cena"


def normal_discount():
    return random.choices([0, 5, 10, 15, 20], weights=[70, 12, 10, 6, 2], k=1)[0]


def normal_payment():
    return random.choices(METODOS_PAGO, weights=[45, 40, 15], k=1)[0]


def normal_transaction_type():
    return random.choices(TIPOS_TRANSACCION, weights=[96, 2, 2], k=1)[0]


def base_transaction_amount(menu_item, cantidad_items):
    subtotal = menu_item.precio * cantidad_items
    service_or_noise = random.uniform(0.95, 1.12)
    return round(subtotal * service_or_noise, 2)


def make_base_record(idx, start_date, end_date):
    fecha_hora = normal_datetime(start_date, end_date)
    item = weighted_choice(MENU, "peso")
    cantidad_items = random.choices([1, 2, 3, 4, 5, 6], weights=[30, 30, 18, 12, 6, 4], k=1)[0]
    monto_bruto = base_transaction_amount(item, cantidad_items)
    descuento_pct = normal_discount()
    tipo_transaccion = normal_transaction_type()
    monto_final = round(monto_bruto * (1 - descuento_pct / 100), 2)
    if tipo_transaccion in {"devolucion", "anulacion"}:
        monto_final = round(min(monto_final, random.uniform(1.0, 18.0)), 2)

    hora_decimal = round(fecha_hora.hour + fecha_hora.minute / 60, 2)
    return {
        "id_transaccion": f"TRX-{idx:06d}",
        "fecha_hora": fecha_hora.strftime("%Y-%m-%d %H:%M:%S"),
        "dia_semana": fecha_hora.weekday(),
        "hora_decimal": hora_decimal,
        "turno": turn_for_hour(fecha_hora.hour),
        "cajero": random.choice(CAJEROS),
        "mesero": random.choice(MESEROS),
        "mesa_canal": random.choice(CANAL_MESA),
        "categoria_producto": item.categoria,
        "producto": item.producto,
        "cantidad_items": cantidad_items,
        "monto_bruto": monto_bruto,
        "descuento_pct": descuento_pct,
        "monto_final": monto_final,
        "metodo_pago": normal_payment(),
        "tipo_transaccion": tipo_transaccion,
        "es_anomalia": 0,
        "tipo_anomalia": "normal",
        "descripcion_anomalia": "Transaccion dentro del patron esperado.",
    }


def apply_anomaly(record, anomaly_type, start_date, end_date):
    record = dict(record)
    record["es_anomalia"] = 1
    record["tipo_anomalia"] = anomaly_type

    if anomaly_type == "descuento_fuera_politica":
        record["descuento_pct"] = random.choice([45, 50, 60, 70, 80])
        record["monto_final"] = round(record["monto_bruto"] * (1 - record["descuento_pct"] / 100), 2)
        record["descripcion_anomalia"] = "Descuento superior a la politica normal del restaurante."

    elif anomaly_type == "venta_fuera_horario":
        fecha_hora = outside_hours_datetime(start_date, end_date)
        record["fecha_hora"] = fecha_hora.strftime("%Y-%m-%d %H:%M:%S")
        record["dia_semana"] = fecha_hora.weekday()
        record["hora_decimal"] = round(fecha_hora.hour + fecha_hora.minute / 60, 2)
        record["turno"] = "fuera_horario"
        record["descripcion_anomalia"] = "Transaccion registrada cuando el restaurante deberia estar cerrado."

    elif anomaly_type == "devolucion_monto_alto":
        record["tipo_transaccion"] = "devolucion"
        record["monto_bruto"] = round(random.uniform(80, 250), 2)
        record["descuento_pct"] = 0
        record["monto_final"] = record["monto_bruto"]
        record["descripcion_anomalia"] = "Devolucion con monto muy superior al comportamiento normal."

    elif anomaly_type == "monto_extremo":
        factor = random.uniform(8, 15)
        record["monto_bruto"] = round(max(record["monto_bruto"] * factor, random.uniform(120, 350)), 2)
        record["descuento_pct"] = random.choice([0, 5, 10])
        record["monto_final"] = round(record["monto_bruto"] * (1 - record["descuento_pct"] / 100), 2)
        record["descripcion_anomalia"] = "Monto final extremadamente alto frente al ticket promedio."

    elif anomaly_type == "monto_inconsistente":
        record["monto_bruto"] = random.choice([0, -5, -12, -25])
        record["descuento_pct"] = 0
        record["monto_final"] = record["monto_bruto"]
        record["descripcion_anomalia"] = "Monto cero o negativo, inconsistente para una venta."

    elif anomaly_type == "efectivo_descuento_alto":
        record["metodo_pago"] = "efectivo"
        record["descuento_pct"] = random.choice([35, 40, 50, 60])
        record["monto_final"] = round(record["monto_bruto"] * (1 - record["descuento_pct"] / 100), 2)
        record["descripcion_anomalia"] = "Venta en efectivo con descuento inusualmente alto."

    elif anomaly_type == "empleado_anulaciones_altas":
        record["cajero"] = "CAJ-003"
        record["mesero"] = "MES-004"
        record["tipo_transaccion"] = "anulacion"
        record["monto_bruto"] = round(random.uniform(25, 120), 2)
        record["descuento_pct"] = 0
        record["monto_final"] = record["monto_bruto"]
        record["descripcion_anomalia"] = "Empleado asociado a anulaciones frecuentes o de monto alto."

    return record


def assign_split(record):
    if record["es_anomalia"]:
        return random.choices(["validacion", "prueba"], weights=[35, 65], k=1)[0]
    return random.choices(["train_normal", "validacion", "prueba"], weights=[70, 15, 15], k=1)[0]


def generate_dataset(total_rows=12000, anomaly_ratio=0.15):
    random.seed(SEED)
    start_date = datetime(2026, 5, 1)
    end_date = datetime(2026, 6, 30)
    anomaly_count = int(total_rows * anomaly_ratio)
    normal_count = total_rows - anomaly_count

    records = []
    for idx in range(1, normal_count + 1):
        records.append(make_base_record(idx, start_date, end_date))

    anomaly_types = [
        "descuento_fuera_politica",
        "venta_fuera_horario",
        "devolucion_monto_alto",
        "monto_extremo",
        "monto_inconsistente",
        "efectivo_descuento_alto",
        "empleado_anulaciones_altas",
    ]
    anomaly_weights = [18, 14, 15, 15, 12, 14, 12]

    for offset in range(anomaly_count):
        idx = normal_count + offset + 1
        base = make_base_record(idx, start_date, end_date)
        anomaly_type = random.choices(anomaly_types, weights=anomaly_weights, k=1)[0]
        records.append(apply_anomaly(base, anomaly_type, start_date, end_date))

    random.shuffle(records)
    for i, record in enumerate(records, start=1):
        record["id_transaccion"] = f"TRX-{i:06d}"
        record["split_sugerido"] = assign_split(record)
    return records


def write_csv(records):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id_transaccion",
        "fecha_hora",
        "dia_semana",
        "hora_decimal",
        "turno",
        "cajero",
        "mesero",
        "mesa_canal",
        "categoria_producto",
        "producto",
        "cantidad_items",
        "monto_bruto",
        "descuento_pct",
        "monto_final",
        "metodo_pago",
        "tipo_transaccion",
        "es_anomalia",
        "tipo_anomalia",
        "descripcion_anomalia",
        "split_sugerido",
    ]
    with OUTPUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    records = generate_dataset()
    write_csv(records)
    print(f"Dataset generado: {OUTPUT}")
    print(f"Filas: {len(records)}")
    print(f"Anomalias: {sum(int(r['es_anomalia']) for r in records)}")


if __name__ == "__main__":
    main()
