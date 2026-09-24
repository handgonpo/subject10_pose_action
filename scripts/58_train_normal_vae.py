from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.vae_model import SequenceVAE, vae_loss


CONFIG_PATH = ROOT / "configs" / "day09_vae.json"
TRAIN_INFO_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_training_info.json"
)
TRAIN_PATH = (
    ROOT / "data" / "day09" / "vae" / "train_normal.csv"
)
MODEL_PATH = (
    ROOT
    / "artifacts"
    / "day09"
    / "vae"
    / "normal_sequence_vae.pt"
)
SCALER_PATH = (
    ROOT
    / "artifacts"
    / "day09"
    / "vae"
    / "normal_sequence_scaler.joblib"
)
META_PATH = (
    ROOT
    / "artifacts"
    / "day09"
    / "vae"
    / "vae_meta.json"
)
LOSS_PATH = (
    ROOT
    / "reports"
    / "day09"
    / "vae"
    / "training_loss.csv"
)

META_COLUMNS = {
    "clip_id",
    "subject",
    "action_id",
    "action_label",
    "take",
    "window_id",
    "source_start_frame",
    "source_end_frame",
    "valid_frames_in_window",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    config = load_json(CONFIG_PATH)
    training_info = load_json(TRAIN_INFO_PATH)
    seed = int(config["random_state"])
    set_seed(seed)

    df = pd.read_csv(TRAIN_PATH)

    if df.empty:
        raise RuntimeError(
            "VAE Train Normal Dataset이 비어 있습니다."
        )

    if set(df["action_label"].astype(str)) != {
        str(config["normal_action_label"])
    }:
        raise RuntimeError(
            "Train Normal에는 Reference Action만 있어야 합니다."
        )

    feature_columns = [
        column
        for column in df.columns
        if column not in META_COLUMNS
    ]

    expected_feature_count = (
        int(training_info["sequence_steps"]) * 38
    )

    if len(feature_columns) != expected_feature_count:
        raise RuntimeError(
            "VAE 입력 Feature 수가 "
            "sequence_steps × 38과 다릅니다."
        )

    X = df[feature_columns].to_numpy(
        dtype=np.float32
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(
        np.float32
    )

    dataset = TensorDataset(
        torch.from_numpy(X_scaled)
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SequenceVAE(
        input_dim=X_scaled.shape[1],
        hidden_dim=int(config["hidden_dim"]),
        latent_dim=int(config["latent_dim"]),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
    )

    epochs = int(config["epochs"])
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        batch_count = 0

        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            reconstruction, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss(
                reconstruction,
                batch,
                mu,
                logvar,
                beta=float(config["beta"]),
            )

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_recon += float(recon_loss.item())
            total_kl += float(kl_loss.item())
            batch_count += 1

        if batch_count == 0:
            raise RuntimeError(
                "VAE Training Batch가 없습니다."
            )

        row = {
            "epoch": epoch,
            "loss": total_loss / batch_count,
            "reconstruction_loss": (
                total_recon / batch_count
            ),
            "kl_loss": total_kl / batch_count,
        }
        history.append(row)

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):
            print(
                f"Epoch {epoch:03d}/{epochs} "
                f"Loss={row['loss']:.6f} "
                f"Recon={row['reconstruction_loss']:.6f} "
                f"KL={row['kl_loss']:.6f}"
            )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    meta = {
        "input_dim": int(X_scaled.shape[1]),
        "hidden_dim": int(config["hidden_dim"]),
        "latent_dim": int(config["latent_dim"]),
        "sequence_steps": int(
            training_info["sequence_steps"]
        ),
        "feature_columns": feature_columns,
        "normal_action_label": str(
            config["normal_action_label"]
        ),
        "train_windows": len(df),
    }

    META_PATH.write_text(
        json.dumps(
            meta,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    LOSS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    pd.DataFrame(history).to_csv(
        LOSS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(f"Device        : {device}")
    print(f"Train windows : {len(df)}")
    print(f"Input dim     : {X_scaled.shape[1]}")
    print(f"Model         : {MODEL_PATH}")
    print(f"Scaler        : {SCALER_PATH}")
    print(f"Meta          : {META_PATH}")


if __name__ == "__main__":
    main()