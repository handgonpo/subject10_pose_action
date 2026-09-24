from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.sequence_utils import (
    build_frame_feature_matrix,
    flatten_sequence,
    resample_sequence,
    sequence_feature_names,
)


CONFIG_PATH = ROOT / "configs" / "day09_vae.json"
TRAIN_INFO_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_training_info.json"
)
TRAIN_SPLIT = (
    ROOT / "data" / "day05" / "splits" / "train.csv"
)
VAL_SPLIT = (
    ROOT / "data" / "day05" / "splits" / "val.csv"
)
OUTPUT_DIR = ROOT / "data" / "day09" / "vae"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def windows_from_clip(
    item,
    config: dict,
    target_steps: int,
):
    keypoint_path = ROOT / str(item["keypoint_csv"])
    frame_df = pd.read_csv(keypoint_path)

    frame_matrix, frame_ids = build_frame_feature_matrix(
        frame_df
    )

    window_size = int(config["window_valid_frames"])
    stride = int(config["window_stride"])

    if stride < 1:
        raise RuntimeError(
            "window_stride는 1 이상이어야 합니다."
        )

    if len(frame_matrix) < window_size:
        return []

    feature_names = sequence_feature_names(target_steps)
    rows = []
    window_id = 0

    for start in range(
        0,
        len(frame_matrix) - window_size + 1,
        stride,
    ):
        end = start + window_size

        window_matrix = frame_matrix[start:end]
        window_frame_ids = frame_ids[start:end]

        sequence = resample_sequence(
            window_matrix,
            window_frame_ids,
            target_steps=target_steps,
        )
        flattened = flatten_sequence(sequence)

        row = {
            "clip_id": item["clip_id"],
            "subject": item["subject"],
            "action_id": item["action_id"],
            "action_label": item["action_label"],
            "take": item["take"],
            "window_id": window_id,
            "source_start_frame": int(
                window_frame_ids[0]
            ),
            "source_end_frame": int(
                window_frame_ids[-1]
            ),
            "valid_frames_in_window": len(
                window_matrix
            ),
        }

        for name, value in zip(feature_names, flattened):
            row[name] = float(value)

        rows.append(row)
        window_id += 1

    return rows


def build_dataset(
    split_path: Path,
    output_path: Path,
    config: dict,
    target_steps: int,
    only_normal: bool,
):
    manifest = pd.read_csv(split_path)

    if only_normal:
        manifest = manifest[
            manifest["action_label"].astype(str)
            == str(config["normal_action_label"])
        ].copy()

    rows = []

    for _, item in manifest.iterrows():
        clip_rows = windows_from_clip(
            item,
            config,
            target_steps,
        )
        rows.extend(clip_rows)

        print(
            f"{item['clip_id']} "
            f"→ {len(clip_rows)} windows"
        )

    if not rows:
        raise RuntimeError(
            f"VAE Window가 없습니다: {output_path}"
        )

    result = pd.DataFrame(rows)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(f"Rows : {len(result)}")
    print(f"Saved: {output_path}")
    print()


def main() -> None:
    config = load_json(CONFIG_PATH)
    training_info = load_json(TRAIN_INFO_PATH)
    target_steps = int(training_info["sequence_steps"])

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    build_dataset(
        TRAIN_SPLIT,
        OUTPUT_DIR / "train_normal.csv",
        config,
        target_steps,
        only_normal=True,
    )

    build_dataset(
        VAL_SPLIT,
        OUTPUT_DIR / "val_all.csv",
        config,
        target_steps,
        only_normal=False,
    )

    val = pd.read_csv(OUTPUT_DIR / "val_all.csv")

    if val["action_label"].nunique() != 6:
        raise RuntimeError(
            "Validation VAE Dataset에 6개 Action이 "
            "모두 있어야 합니다."
        )

    print("Day09 VAE Window Dataset: READY")


if __name__ == "__main__":
    main()