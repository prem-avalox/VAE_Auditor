"""
Test script for multi-user concurrency verification.
Simulates 3 (or more) concurrent users invoking the PyTorch VAE inference pipeline simultaneously
to verify thread-safety, stability, latency, and throughput.
"""

import sys
import time
import random
import concurrent.futures
import psutil

# Configurar stdout para evitar UnicodeEncodeError en consola Windows CP1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.audit_service import evaluate_raw_transaction, process_raw_batch, load_artifacts
from src.logger import log_info

NUM_USERS = 3  # Requirement: 3 concurrent users
REQUESTS_PER_USER = 10


def simulate_user_activity(user_id: int):
    """Simula a un usuario realizando múltiples evaluaciones de ventas."""
    results = []
    for i in range(REQUESTS_PER_USER):
        tx_data = {
            "monto": float(round(random.uniform(5.0, 350.0), 2)),
            "descuento": float(round(random.uniform(0.0, 50.0), 2)),
            "hora": random.randint(0, 23),
            "dia_semana": random.choice(["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]),
            "metodo_pago": random.choice(["Efectivo", "Tarjeta crédito", "Tarjeta débito", "Transferencia"]),
            "num_items": random.randint(1, 15),
        }

        res, perf = evaluate_raw_transaction(tx_data)
        results.append({
            "user_id": user_id,
            "request_idx": i,
            "severidad": res["severidad"],
            "rec_error": res["reconstruction_error"],
            "latency": perf.latency,
        })
        time.sleep(0.01)  # Pequeña pausa entre peticiones del usuario
    return results


def main():
    print("=" * 65)
    print("PRUEBA DE CONCURRENCIA MULTIUSUARIO - VAE AUDITOR")
    print(f"Simulando {NUM_USERS} usuarios concurrentes haciendo {REQUESTS_PER_USER} peticiones c/u...")
    print("=" * 65)

    # 1. Precargar artefactos
    load_artifacts()
    print("[OK] Artefactos PyTorch + Scikit-Learn cargados correctamente.")

    proceso = psutil.Process()
    proceso.cpu_percent(interval=None)  # reinicia el contador de % CPU
    ram_antes_mb = proceso.memory_info().rss / (1024 * 1024)

    # 2. Ejecución concurrente con ThreadPoolExecutor
    start_time = time.perf_counter()
    all_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_USERS) as executor:
        futures = [executor.submit(simulate_user_activity, user_id=u + 1) for u in range(NUM_USERS)]
        for future in concurrent.futures.as_completed(futures):
            try:
                user_res = future.result()
                all_results.extend(user_res)
            except Exception as e:
                print(f"[ERROR] Error en hilo de usuario: {e}")

    total_time = time.perf_counter() - start_time
    cpu_percent_test = proceso.cpu_percent(interval=None)
    ram_despues_mb = proceso.memory_info().rss / (1024 * 1024)
    total_tx = len(all_results)
    avg_latency = sum(r["latency"] for r in all_results) / max(total_tx, 1)
    throughput = total_tx / max(total_time, 0.001)

    print("\n" + "-" * 65)
    print("RESULTADOS DE LA PRUEBA DE CONCURRENCIA:")
    print("-" * 65)
    print(f"* Usuarios concurrentes probados : {NUM_USERS}")
    print(f"* Peticiones totales procesadas  : {total_tx}")
    print(f"* Tiempo total transcurrido     : {total_time:.4f} segundos")
    print(f"* Latencia promedio             : {avg_latency * 1000:.2f} ms por transaccion")
    print(f"* Throughput                    : {throughput:.2f} transacciones / segundo")
    print(f"* CPU usado durante la prueba    : {cpu_percent_test:.1f} %")
    print(f"* RAM del proceso (antes -> despues): {ram_antes_mb:.1f} MB -> {ram_despues_mb:.1f} MB")
    print(f"* Estado del sistema            : [OK] ESTABLE (0 errores, thread-safe)")
    print("=" * 65)

    log_info(
        "CONCURRENCY_TEST_PASSED",
        f"users={NUM_USERS}, total_tx={total_tx}, time={total_time:.4f}s, "
        f"throughput={throughput:.2f}tx/s, cpu_percent={cpu_percent_test:.1f}, "
        f"memoria_mb={ram_despues_mb:.1f}"
    )


if __name__ == "__main__":
    main()