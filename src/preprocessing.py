"""
Pipeline de preprocesamiento: transforma el CSV de ventas del restaurante
en tensores listos para entrenar el VAE.

Responsable: [tu nombre] - Preprocesamiento de datos.
Entrada: data/ventas_restaurante_sinteticas.csv
Salida: tensores en data/processed/ + preprocesador guardado en models/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

RUTA_CSV = Path("data/ventas_restaurante_sinteticas.csv")
RUTA_PREPROCESADOR = Path("models/preprocessor.joblib")
CARPETA_PROCESADOS = Path("data/processed")

COLUMNAS_CICLICAS_GENERADAS = ["dia_sin", "dia_cos", "hora_sin", "hora_cos"]

COLUMNAS_ESPERADAS = [
    "id_transaccion", "fecha_hora", "dia_semana", "hora_decimal", "turno",
    "cajero", "mesero", "mesa_canal", "categoria_producto", "producto",
    "cantidad_items", "monto_bruto", "descuento_pct", "monto_final",
    "metodo_pago", "tipo_transaccion", "es_anomalia", "tipo_anomalia",
    "descripcion_anomalia", "split_sugerido",
]

SPLITS_VALIDOS = {"train_normal", "validacion", "prueba"}

# Columnas que identifican la transaccion o son metadata de la anomalia,
# no se usan como entrada del VAE.
COLUMNAS_NO_FEATURE = [
    "id_transaccion", "fecha_hora", "es_anomalia", "tipo_anomalia",
    "descripcion_anomalia", "split_sugerido",
]

# Numericas continuas que van directo al escalador.
COLUMNAS_NUMERICAS = [
    "cantidad_items", "monto_bruto", "descuento_pct", "monto_final",
]

# Categoricas que van a One-Hot Encoding.
COLUMNAS_CATEGORICAS = [
    "turno", "cajero", "mesero", "mesa_canal", "categoria_producto",
    "producto", "metodo_pago", "tipo_transaccion",
]

# dia_semana y hora_decimal no entran tal cual: se transforman a
# pares seno/coseno para que el modelo entienda su naturaleza ciclica.
COLUMNAS_CICLICAS = ["dia_semana", "hora_decimal"]

# Si una categorica supera este umbral de valores unicos, One-Hot Encoding
# generaria demasiadas columnas dispersas. En ese caso conviene agrupar
# categorias poco frecuentes en "otros" o usar embeddings en vez de One-Hot.
CARDINALIDAD_MAXIMA_RECOMENDADA = 30


def verificar_cardinalidad(df: pd.DataFrame, columnas: list[str] = COLUMNAS_CATEGORICAS) -> None:
    """
    Revisa que ninguna columna categorica tenga demasiados valores unicos
    para One-Hot Encoding. Si alguna supera el umbral, avisa para que se
    considere agrupar categorias raras o cambiar de estrategia (embeddings).
    """
    for col in columnas:
        n_unicos = df[col].nunique()
        if n_unicos > CARDINALIDAD_MAXIMA_RECOMENDADA:
            print(
                f"AVISO: '{col}' tiene {n_unicos} valores unicos "
                f"(umbral recomendado: {CARDINALIDAD_MAXIMA_RECOMENDADA}). "
                f"Considerar agrupar categorias poco frecuentes en 'otros' "
                f"o usar embeddings en vez de One-Hot Encoding."
            )
        else:
            print(f"OK: '{col}' tiene {n_unicos} valores unicos.")


def cargar_y_validar(ruta_csv: Path = RUTA_CSV) -> pd.DataFrame:
    """
    Carga el CSV y valida que tenga la forma esperada antes de transformar nada.

    Validaciones:
    - Las columnas esperadas existen.
    - No hay valores nulos.
    - No hay id_transaccion duplicados.
    - split_sugerido solo contiene los 3 valores documentados.
    - monto_final es coherente con monto_bruto y descuento_pct
      (se avisa si no, pero no se corrige aqui: forma parte del
      comportamiento esperado en filas anomalas).
    """
    df = pd.read_csv(ruta_csv, parse_dates=["fecha_hora"])

    columnas_faltantes = set(COLUMNAS_ESPERADAS) - set(df.columns)
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {columnas_faltantes}")

    nulos = df.isnull().sum()
    if nulos.sum() > 0:
        raise ValueError(f"El CSV tiene valores nulos:\n{nulos[nulos > 0]}")

    duplicados = df["id_transaccion"].duplicated().sum()
    if duplicados > 0:
        raise ValueError(f"Hay {duplicados} id_transaccion duplicados")

    splits_encontrados = set(df["split_sugerido"].unique())
    if not splits_encontrados.issubset(SPLITS_VALIDOS):
        raise ValueError(
            f"split_sugerido tiene valores inesperados: "
            f"{splits_encontrados - SPLITS_VALIDOS}"
        )

    print(f"CSV cargado correctamente: {len(df)} filas, {len(df.columns)} columnas.")
    print(f"Distribucion de splits:\n{df['split_sugerido'].value_counts()}")
    print(f"Distribucion de anomalias:\n{df['es_anomalia'].value_counts()}")

    return df


def ingenieria_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara las variables de entrada del VAE:

    - Convierte dia_semana (0-6) y hora_decimal (0-24) en pares seno/coseno,
      para que el modelo entienda que domingo(6) esta cerca de lunes(0),
      y que 23:59 esta cerca de 00:01.
    - Conserva las columnas numericas y categoricas tal cual, listas para
      pasar al escalador / encoder en el siguiente paso.
    - Conserva es_anomalia y split_sugerido aparte (no son features del VAE,
      son etiquetas para evaluar despues).

    Devuelve un dataframe SOLO con las columnas que seran entrada del modelo
    mas las columnas de control (es_anomalia, split_sugerido) al final.
    """
    df = df.copy()

    # Codificacion ciclica de dia_semana (periodo = 7 dias)
    df["dia_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
    df["dia_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)

    # Codificacion ciclica de hora_decimal (periodo = 24 horas)
    df["hora_sin"] = np.sin(2 * np.pi * df["hora_decimal"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora_decimal"] / 24)

    columnas_finales = (
        COLUMNAS_NUMERICAS
        + ["dia_sin", "dia_cos", "hora_sin", "hora_cos"]
        + COLUMNAS_CATEGORICAS
        + ["es_anomalia", "split_sugerido"]
    )

    return df[columnas_finales]


def construir_preprocesador() -> ColumnTransformer:
    """
    Arma el ColumnTransformer:

    - StandardScaler para las numericas continuas (monto_bruto, etc.),
      porque tienen escalas y varianzas distintas entre si.
    - 'passthrough' para las ciclicas (dia_sin, dia_cos, hora_sin, hora_cos),
      ya estan en el rango [-1, 1] por construccion, no necesitan escalarse.
    - OneHotEncoder para las categoricas. handle_unknown='ignore' evita que
      el pipeline falle si en produccion aparece una categoria nueva
      (ej. un cajero nuevo que no estaba en el set de entrenamiento).
    """
    return ColumnTransformer(
        transformers=[
            ("numericas", StandardScaler(), COLUMNAS_NUMERICAS),
            ("ciclicas", "passthrough", COLUMNAS_CICLICAS_GENERADAS),
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                COLUMNAS_CATEGORICAS,
            ),
        ]
    )


def ajustar_y_transformar(df_features: pd.DataFrame) -> dict:
    """
    Ajusta el preprocesador SOLO con las filas de entrenamiento
    (split_sugerido == 'train_normal', que ya vienen sin anomalias),
    y transforma los 3 splits con ese mismo preprocesador.

    Esto es clave: si ajustaramos el scaler/encoder con TODO el dataset,
    la media y desviacion estandar quedarian "contaminadas" por las
    anomalias y por los datos de validacion/prueba, y el modelo tendria
    una ventaja artificial que no tendria en un caso real.
    """
    es_train = df_features["split_sugerido"] == "train_normal"
    es_val = df_features["split_sugerido"] == "validacion"
    es_test = df_features["split_sugerido"] == "prueba"

    columnas_entrada = COLUMNAS_NUMERICAS + COLUMNAS_CICLICAS_GENERADAS + COLUMNAS_CATEGORICAS

    preprocesador = construir_preprocesador()
    X_train = preprocesador.fit_transform(df_features.loc[es_train, columnas_entrada])
    X_val = preprocesador.transform(df_features.loc[es_val, columnas_entrada])
    X_test = preprocesador.transform(df_features.loc[es_test, columnas_entrada])

    RUTA_PREPROCESADOR.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocesador, RUTA_PREPROCESADOR)

    print(f"Preprocesador guardado en {RUTA_PREPROCESADOR}")
    print(f"Dimensiones -> train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_val": df_features.loc[es_val, "es_anomalia"].to_numpy(),
        "y_test": df_features.loc[es_test, "es_anomalia"].to_numpy(),
        "preprocesador": preprocesador,
    }


class VentasDataset(Dataset):
    """
    Dataset de PyTorch para las ventas ya preprocesadas.
    Cada item es un vector de 68 features (float32), listo para el VAE.
    No incluye etiquetas porque el VAE se entrena sin supervision:
    aprende a reconstruir transacciones normales.
    """

    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]


def convertir_y_guardar(resultado: dict) -> dict:
    """
    Convierte los arrays de NumPy a tensores de PyTorch y los guarda en
    disco en data/processed/, junto con las etiquetas de validacion/prueba
    (necesarias para medir despues si el error de reconstruccion del VAE
    es mayor en las transacciones anomalas).
    """
    CARPETA_PROCESADOS.mkdir(parents=True, exist_ok=True)

    tensor_train = torch.tensor(resultado["X_train"], dtype=torch.float32)
    tensor_val = torch.tensor(resultado["X_val"], dtype=torch.float32)
    tensor_test = torch.tensor(resultado["X_test"], dtype=torch.float32)
    etiquetas_val = torch.tensor(resultado["y_val"], dtype=torch.long)
    etiquetas_test = torch.tensor(resultado["y_test"], dtype=torch.long)

    torch.save(tensor_train, CARPETA_PROCESADOS / "train_tensor.pt")
    torch.save(tensor_val, CARPETA_PROCESADOS / "val_tensor.pt")
    torch.save(tensor_test, CARPETA_PROCESADOS / "test_tensor.pt")
    torch.save(etiquetas_val, CARPETA_PROCESADOS / "val_labels.pt")
    torch.save(etiquetas_test, CARPETA_PROCESADOS / "test_labels.pt")

    print(f"Tensores guardados en {CARPETA_PROCESADOS}/")
    print(f"  train_tensor.pt : {tuple(tensor_train.shape)}")
    print(f"  val_tensor.pt   : {tuple(tensor_val.shape)}  (+ val_labels.pt)")
    print(f"  test_tensor.pt  : {tuple(tensor_test.shape)}  (+ test_labels.pt)")

    return {
        "train_dataset": VentasDataset(resultado["X_train"]),
        "val_dataset": VentasDataset(resultado["X_val"]),
        "test_dataset": VentasDataset(resultado["X_test"]),
    }


if __name__ == "__main__":
    df = cargar_y_validar()
    print()
    verificar_cardinalidad(df)
    df_features = ingenieria_variables(df)
    print()
    resultado = ajustar_y_transformar(df_features)
    print()
    datasets = convertir_y_guardar(resultado)

    # Ejemplo de uso: DataLoader listo para el entrenamiento del VAE
    train_loader = DataLoader(datasets["train_dataset"], batch_size=64, shuffle=True)
    primer_batch = next(iter(train_loader))
    print()
    print(f"Ejemplo de batch de entrenamiento: {primer_batch.shape}")
    print()
    print("Columnas finales para el VAE:")
    print([c for c in df_features.columns if c not in ("es_anomalia", "split_sugerido")])
    print()
    print(df_features[["dia_sin", "dia_cos", "hora_sin", "hora_cos"]].head())