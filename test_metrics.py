from src.metrics import Metrics
import time

m = Metrics()

m.start()

time.sleep(2)

resultado = m.stop(100)

print(resultado)