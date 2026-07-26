from dataclasses import dataclass
import time
import psutil


@dataclass
class MetricsResult:
    total_transactions: int
    total_time: float
    latency: float
    throughput: float
    cpu_percent: float
    memory_mb: float


class Metrics:

    def __init__(self):
        self._start = None
        self._process = psutil.Process()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def start(self):
        self._start = time.perf_counter()
        # Primera llamada "en blanco": psutil mide el % de CPU acumulado
        # DESDE la última llamada, así que esta primera lectura solo sirve
        # para poner en cero el contador antes de medir el tramo real.
        self._process.cpu_percent(interval=None)

    def stop(self, total_transactions: int):

        if self._start is None:
            raise RuntimeError("Debe llamar a start() antes de stop().")

        total_time = time.perf_counter() - self._start

        latency = (
            total_time / total_transactions
            if total_transactions > 0
            else 0
        )

        throughput = (
            total_transactions / total_time
            if total_time > 0
            else 0
        )

        # Uso real de CPU y RAM del proceso de Python que corre el modelo,
        # no del sistema operativo completo (así la cifra representa
        # específicamente lo que consume la inferencia, no otros programas
        # abiertos en la misma máquina).
        cpu_percent = self._process.cpu_percent(interval=None)
        memory_mb = self._process.memory_info().rss / (1024 * 1024)

        return MetricsResult(
            total_transactions=total_transactions,
            total_time=total_time,
            latency=latency,
            throughput=throughput,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
        )