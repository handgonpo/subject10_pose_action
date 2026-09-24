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


CONFIG_PATH = ROOT / "configs" / "day09_vae.json"
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
ERROR_PATH = (
    ROOT
    / "reports"
    / "day09"
    / "vae"
    / "validation_normal_errors.csv"
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    config = load_json(CONFIG_PATH)
    meta = load_json(META_PATH)
    val = pd.read_csv(VAL_PATH)

    normal_label = str(config["normal_action_label"])
    normal = val[
        val["action_label"].astype(str) == normal_label
    ].copy()

    if normal.empty:
        raise RuntimeError(
            "Validation Normal Reference Window가 없습니다."
        )

    feature_columns = meta["feature_columns"]
    X = normal[feature_columns].to_numpy(
        dtype=np.float32
    )

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
    percentile = float(config["threshold_percentile"])
    threshold = float(
        np.percentile(errors, percentile)
    )

    output = normal[
        [
            "clip_id",
            "subject",
            "action_label",
            "window_id",
            "source_start_frame",
            "source_end_frame",
        ]
    ].copy()
    output["reconstruction_error"] = errors

    ERROR_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.to_csv(
        ERROR_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    threshold_info = {
        "normal_action_label": normal_label,
        "threshold_percentile": percentile,
        "threshold": threshold,
        "normal_validation_windows": len(normal),
        "mean_normal_error": float(errors.mean()),
        "median_normal_error": float(np.median(errors)),
    }

    THRESHOLD_PATH.write_text(
        json.dumps(
            threshold_info,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Normal Action : {normal_label}")
    print(f"Val Windows   : {len(normal)}")
    print(f"Mean Error    : {errors.mean():.6f}")
    print(
        f"Threshold P{percentile:g}: "
        f"{threshold:.6f}"
    )
    print(f"Saved: {THRESHOLD_PATH}")
    print(f"Saved: {ERROR_PATH}")


if __name__ == "__main__":
    main()