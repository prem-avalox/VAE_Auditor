"""
Locustfile for performance and load testing of the VAE Auditor FastAPI backend.
Simulates concurrent user load hitting /evaluate, /metrics, /health, and /logs endpoints.

Usage:
    locust -f locustfile.py --host=http://localhost:8000
"""

from locust import HttpUser, task, between
import random

class VAEAuditorUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def evaluate_transaction(self):
        payload = {
            "monto": float(round(random.uniform(5.0, 300.0), 2)),
            "descuento": float(round(random.uniform(0.0, 40.0), 2)),
            "hora": random.randint(0, 23),
            "dia_semana": random.choice(["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]),
            "metodo_pago": random.choice(["Efectivo", "Tarjeta crédito", "Tarjeta débito", "Transferencia"]),
            "num_items": random.randint(1, 10),
        }
        self.client.post("/evaluate", json=payload)

    @task(2)
    def check_health(self):
        self.client.get("/health")

    @task(1)
    def get_metrics(self):
        self.client.get("/metrics")

    @task(1)
    def get_logs(self):
        self.client.get("/logs?lines=20")
