"""Entrenamiento y scoring del VAE para el auditor de ventas.

Contrato de la parte 3:
- Entrena exclusivamente con data/processed/train_tensor.pt.
- Guarda el state_dict en models/vae_model.pt.
- Genera un error MSE determinista por transaccion de validacion y prueba.

Ejecutar desde cualquier directorio con:
    python src/train_vae.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_PATH = PROJECT_ROOT / "data" / "ventas_restaurante_sinteticas.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_TENSOR_PATH = PROCESSED_DIR / "train_tensor.pt"
VAL_TENSOR_PATH = PROCESSED_DIR / "val_tensor.pt"
TEST_TENSOR_PATH = PROCESSED_DIR / "test_tensor.pt"
MODEL_PATH = MODELS_DIR / "vae_model.pt"
MODEL_CONFIG_PATH = MODELS_DIR / "vae_model_config.json"
ERRORS_PATH = REPORTS_DIR / "reconstruction_errors.csv"
HISTORY_PATH = REPORTS_DIR / "vae_training_history.csv"

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_LATENT_DIM = 8
DEFAULT_BETA = 0.001
DEFAULT_SEED = 42


class VAE(nn.Module):
    """VAE denso con la arquitectura minima definida para el proyecto."""

    def __init__(self, input_dim: int, latent_dim: int = DEFAULT_LATENT_DIM):
        super().__init__()
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("input_dim y latent_dim deben ser positivos")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(32, latent_dim)
        self.logvar_layer = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def reconstruct_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruye usando mu, sin muestreo aleatorio durante inferencia."""
        mu, _ = self.encode(x)
        return self.decode(mu)


def vae_loss_components(
    x: torch.Tensor,
    reconstruction: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = DEFAULT_BETA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Devuelve perdida total, MSE y KL promedio por observacion.

    La divergencia KL se suma sobre las dimensiones latentes y luego se
    promedia sobre el lote, que corresponde al KL de cada observacion.
    """
    reconstruction_loss = F.mse_loss(reconstruction, x, reduction="mean")
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(), dim=1
    ).mean()
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss


def vae_loss(
    x: torch.Tensor,
    reconstruction: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = DEFAULT_BETA,
) -> torch.Tensor:
    """Interfaz simple de la perdida requerida: MSE + beta * KL."""
    return vae_loss_components(x, reconstruction, mu, logvar, beta)[0]


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_feature_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} no es un tensor de PyTorch")
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} debe tener forma (filas, features), no {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} debe ser float, no {tensor.dtype}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contiene NaN o valores infinitos")


def load_tensor(path: Path, name: str) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    validate_feature_tensor(tensor, name)
    return tensor.float().contiguous()


def verify_no_training_leakage(expected_rows: int) -> None:
    """Contrasta el tensor de train con la fuente y exige cero anomalias."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"No se puede auditar el split: falta {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH, usecols=["split_sugerido", "es_anomalia"])
    train_rows = df["split_sugerido"].eq("train_normal")
    anomaly_count = int(df.loc[train_rows, "es_anomalia"].astype(int).sum())
    source_rows = int(train_rows.sum())
    if anomaly_count != 0:
        raise ValueError(
            f"Data leakage: train_normal contiene {anomaly_count} anomalias"
        )
    if source_rows != expected_rows:
        raise ValueError(
            "Desalineacion entre CSV y train_tensor.pt: "
            f"{source_rows} filas normales frente a {expected_rows} filas del tensor"
        )


def train_vae(
    model: VAE,
    train_tensor: torch.Tensor,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LEARNING_RATE,
    beta: float = DEFAULT_BETA,
    seed: int = DEFAULT_SEED,
    device: torch.device | None = None,
) -> tuple[VAE, list[dict[str, float | int]]]:
    """Entrena el VAE y devuelve el modelo junto con el historial por epoca."""
    validate_feature_tensor(train_tensor, "train_tensor")
    if epochs <= 0 or batch_size <= 0 or lr <= 0 or beta < 0:
        raise ValueError("Hiperparametros invalidos")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_weighted = 0.0
        mse_weighted = 0.0
        kl_weighted = 0.0
        samples_seen = 0

        for (batch,) in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar = model(batch)
            loss, mse, kl = vae_loss_components(
                batch, reconstruction, mu, logvar, beta
            )
            loss.backward()
            optimizer.step()

            batch_size_actual = batch.shape[0]
            samples_seen += batch_size_actual
            total_weighted += loss.item() * batch_size_actual
            mse_weighted += mse.item() * batch_size_actual
            kl_weighted += kl.item() * batch_size_actual

        epoch_metrics = {
            "epoch": epoch,
            "loss": total_weighted / samples_seen,
            "reconstruction_mse": mse_weighted / samples_seen,
            "kl_loss": kl_weighted / samples_seen,
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch:02d}/{epochs} - loss={epoch_metrics['loss']:.6f} "
            f"mse={epoch_metrics['reconstruction_mse']:.6f} "
            f"kl={epoch_metrics['kl_loss']:.6f}"
        )

    return model, history


def reconstruction_errors(
    model: VAE, tensor: torch.Tensor, batch_size: int = 1024
) -> np.ndarray:
    """Calcula MSE por fila de forma determinista y eficiente por lotes."""
    validate_feature_tensor(tensor, "tensor de scoring")
    if tensor.shape[1] != model.input_dim:
        raise ValueError(
            f"El modelo espera {model.input_dim} features y recibio {tensor.shape[1]}"
        )

    device = next(model.parameters()).device
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    model.eval()
    all_errors: list[torch.Tensor] = []
    with torch.inference_mode():
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction = model.reconstruct_deterministic(batch)
            all_errors.append(F.mse_loss(reconstruction, batch, reduction="none").mean(dim=1).cpu())
    return torch.cat(all_errors).numpy()


def save_model(model: VAE, config: dict[str, int | float | str]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    MODEL_CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_vae(
    model_path: Path = MODEL_PATH,
    config_path: Path = MODEL_CONFIG_PATH,
    device: torch.device | str = "cpu",
) -> VAE:
    """Carga el modelo entrenado sin tener que conocer su input_dim externamente."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = VAE(
        input_dim=int(config["input_dim"]), latent_dim=int(config["latent_dim"])
    )
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def build_error_report(
    val_errors: np.ndarray, test_errors: np.ndarray
) -> pd.DataFrame:
    """Une los scores con IDs y campos necesarios para la evaluacion posterior."""
    usecols = [
        "id_transaccion",
        "split_sugerido",
        "es_anomalia",
        "tipo_anomalia",
        "monto_final",
    ]
    source = pd.read_csv(DATASET_PATH, usecols=usecols)
    frames = []
    for split, errors in (("validacion", val_errors), ("prueba", test_errors)):
        metadata = source.loc[source["split_sugerido"].eq(split)].copy()
        if len(metadata) != len(errors):
            raise ValueError(
                f"Desalineacion en {split}: {len(metadata)} filas y {len(errors)} errores"
            )
        metadata.insert(1, "indice_split", np.arange(len(metadata)))
        metadata.insert(2, "split", split)
        metadata["reconstruction_error"] = errors
        frames.append(metadata)
    return pd.concat(frames, ignore_index=True)


def run_training(args: argparse.Namespace) -> None:
    set_reproducibility(args.seed)
    train_tensor = load_tensor(TRAIN_TENSOR_PATH, "train_tensor")
    verify_no_training_leakage(len(train_tensor))

    input_dim = train_tensor.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    print(f"Entrenamiento auditado: {tuple(train_tensor.shape)}, cero anomalias")
    model = VAE(input_dim=input_dim, latent_dim=args.latent_dim)
    model, history = train_vae(
        model,
        train_tensor,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        beta=args.beta,
        seed=args.seed,
        device=device,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config: dict[str, int | float | str] = {
        "input_dim": input_dim,
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "seed": args.seed,
        "architecture": "input-64-32-(mu,logvar)-latent-32-64-input",
        "training_source": "data/processed/train_tensor.pt",
        "scoring": "deterministic_mse_using_latent_mu",
    }
    save_model(model, config)
    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

    # Validacion y prueba se leen solo despues de terminar el entrenamiento.
    val_tensor = load_tensor(VAL_TENSOR_PATH, "val_tensor")
    test_tensor = load_tensor(TEST_TENSOR_PATH, "test_tensor")
    if val_tensor.shape[1] != input_dim or test_tensor.shape[1] != input_dim:
        raise ValueError("Train, validacion y prueba no tienen el mismo numero de features")

    val_errors = reconstruction_errors(model, val_tensor)
    test_errors = reconstruction_errors(model, test_tensor)
    report = build_error_report(val_errors, test_errors)
    report.to_csv(ERRORS_PATH, index=False)

    # Prueba inmediata de compatibilidad del artefacto guardado.
    loaded_model = load_vae(device="cpu")
    with torch.inference_mode():
        smoke_output = loaded_model.reconstruct_deterministic(train_tensor[:2])
    if smoke_output.shape != train_tensor[:2].shape:
        raise RuntimeError("El modelo guardado no supero la prueba de recarga")

    print(f"Modelo guardado y verificado: {MODEL_PATH}")
    print(f"Configuracion: {MODEL_CONFIG_PATH}")
    print(f"Historial: {HISTORY_PATH}")
    print(f"Errores de reconstruccion: {ERRORS_PATH} ({len(report)} filas)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el VAE del auditor de ventas")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
