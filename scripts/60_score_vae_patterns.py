from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.vae_model import SequenceVAE


VAL_PATH = ROOT / "data" / "day09" / "vae" / "val_all.csv"
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
THRESHOLD_PATH = (
    ROOT
    / "artifacts"
    / "day09"
    / "vae"
    / "threshold.json"
)
WINDOW_OUTPUT = (
    ROOT
    / "reports"
    / "day09"
    / "vae"
    / "validation_all_window_scores.csv"
)
SUMMARY_OUTPUT = (
    ROOT
    / "reports"
    / "day09"
    / "vae"
    / "action_error_summary.csv"
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    meta = load_json(META_PATH)
    threshold_info = load_json(THRESHOLD_PATH)
    threshold = float(threshold_info["threshold"])

    df = pd.read_csv(VAL_PATH)
    feature_columns = meta["feature_columns"]
    X = df[feature_columns].to_numpy(dtype=np.float32)

    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X).astype(np.float32)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SequenceVAE(
        input_dim=int(meta["input_dim"]),
        hidden_dim=int(meta["hidden_dim"]),
        latent_dim=int(meta["latent_dim"]),
    ).to(device)

    state = torch.load(
        MODEL_PATH,
        map_location=device,
    )
    model.load_state_dict(state)
    model.eval()

    tensor = torch.from_numpy(X_scaled).to(device)

    with torch.no_grad():
        mu, _ = model.encode(tensor)
        reconstruction = model.decode(mu)
        errors = torch.mean(
            (reconstruction - tensor) ** 2,
            dim=1,
        )

    errors = errors.detach().cpu().numpy()

    output = df[
        [
            "clip_id",
            "subject",
            "action_id",
            "action_label",
            "window_id",
            "source_start_frame",
            "source_end_frame",
        ]
    ].copy()
    output["reconstruction_error"] = errors
    output["threshold"] = threshold
    output["anomaly_candidate"] = (
        output["reconstruction_error"] > threshold
    )

    WINDOW_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.to_csv(
        WINDOW_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        output.groupby("action_label")
        .agg(
            windows=("window_id", "count"),
            mean_error=(
                "reconstruction_error",
                "mean",
            ),
            median_error=(
                "reconstruction_error",
                "median",
            ),
            max_error=(
                "reconstruction_error",
                "max",
            ),
            anomaly_candidate_ratio=(
                "anomaly_candidate",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "mean_error",
            ascending=False,
        )
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Threshold: {threshold:.6f}")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Window Scores: {WINDOW_OUTPUT}")
    print(f"Summary      : {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()