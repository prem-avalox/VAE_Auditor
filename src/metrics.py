from dataclasses import dataclass
import time


@dataclass
class MetricsResult:
    total_transactions: int
    total_time: float
    latency: float
    throughput: float


class Metrics:

    def __init__(self):
        self._start = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def start(self):
        self._start = time.perf_counter()

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

        return MetricsResult(
            total_transactions=total_transactions,
            total_time=total_time,
            latency=latency,
            throughput=throughput,
        )