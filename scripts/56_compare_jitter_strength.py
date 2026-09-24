from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.augmentation_utils import jitter_dataframe
from src.sequence_utils import (
    build_frame_feature_matrix,
    flatten_sequence,
    resample_sequence,
)


TRAIN_SPLIT = (
    ROOT / "data" / "day05" / "splits" / "train.csv"
)
VAL_PATH = (
    ROOT / "data" / "day06" / "sequences" / "val.csv"
)
FEATURE_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_feature_columns.json"
)
TRAIN_INFO_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_training_info.json"
)
AUG_CONFIG_PATH = (
    ROOT / "configs" / "day09_augmentation.json"
)
OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day09"
    / "experiments"
    / "jitter_strength_comparison.csv"
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_sequence(
    frame_df,
    target_steps: int,
    min_valid_frames: int,
):
    matrix, frame_ids = build_frame_feature_matrix(frame_df)

    if len(matrix) < min_valid_frames:
        return None

    sequence = resample_sequence(
        matrix,
        frame_ids,
        target_steps=target_steps,
    )
    return flatten_sequence(sequence)


def main() -> None:
    train_manifest = pd.read_csv(TRAIN_SPLIT)
    val = pd.read_csv(VAL_PATH)
    feature_columns = load_json(FEATURE_PATH)
    training_info = load_json(TRAIN_INFO_PATH)
    aug_config = load_json(AUG_CONFIG_PATH)

    target_steps = int(training_info["sequence_steps"])
    min_valid_frames = int(
        training_info["min_valid_frames"]
    )
    seed = int(aug_config["random_state"])
    ratios = [
        float(value)
        for value in aug_config["jitter_ratios_compare"]
    ]

    X_val = val[feature_columns].to_numpy(dtype=float)
    y_val = val["action_label"].astype(str).to_numpy()

    rows = []

    for ratio in ratios:
        X_train = []
        y_train = []

        for index, item in train_manifest.iterrows():
            keypoint_path = ROOT / str(item["keypoint_csv"])
            original_df = pd.read_csv(keypoint_path)

            original_seq = build_sequence(
                original_df,
                target_steps,
                min_valid_frames,
            )

            jittered_df = jitter_dataframe(
                original_df,
                jitter_ratio=ratio,
                random_state=seed + int(index),
            )
            jitter_seq = build_sequence(
                jittered_df,
                target_steps,
                min_valid_frames,
            )

            if original_seq is None or jitter_seq is None:
                raise RuntimeError(
                    f"Sequence 생성 실패: {item['clip_id']}"
                )

            X_train.extend([original_seq, jitter_seq])
            y_train.extend(
                [item["action_label"], item["action_label"]]
            )

        model = RandomForestClassifier(
            n_estimators=int(
                training_info["n_estimators"]
            ),
            max_depth=training_info["max_depth"],
            random_state=int(
                training_info["random_state"]
            ),
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        rows.append(
            {
                "jitter_ratio": ratio,
                "train_rows": len(X_train),
                "accuracy": accuracy_score(y_val, pred),
                "macro_f1": f1_score(
                    y_val,
                    pred,
                    average="macro",
                    zero_division=0,
                ),
            }
        )

    result = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(result.to_string(index=False))
    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()