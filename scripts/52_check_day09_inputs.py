from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.sequence_utils import (
    build_frame_feature_matrix,
    sequence_feature_names,
)


SPLIT_DIR = ROOT / "data" / "day05" / "splits"
SEQUENCE_DIR = ROOT / "data" / "day06" / "sequences"

MODEL_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_action_baseline.joblib"
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
VAE_CONFIG_PATH = (
    ROOT / "configs" / "day09_vae.json"
)
OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day09"
    / "tables"
    / "day09_input_check.csv"
)

EXPECTED = {
    "train": ("S01", 6),
    "val": ("S02", 6),
    "test": ("S03", 6),
}

EXPECTED_ACTIONS = {
    "bend_return",
    "leg_raise_lower",
    "walk_turn_walk",
    "drink_return",
    "sit_stand",
    "wave",
}

REQUIRED_TRAIN_INFO_KEYS = {
    "classes",
    "feature_count",
    "sequence_steps",
    "min_valid_frames",
    "n_estimators",
    "max_depth",
    "random_state",
}

REQUIRED_AUG_KEYS = {
    "jitter_ratio",
    "jitter_ratios_compare",
    "random_state",
}

REQUIRED_VAE_KEYS = {
    "normal_action_label",
    "window_valid_frames",
    "window_stride",
    "hidden_dim",
    "latent_dim",
    "epochs",
    "batch_size",
    "learning_rate",
    "beta",
    "threshold_percentile",
    "thresholds_compare",
    "random_state",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def check_required_keys(
    data: dict,
    required: set[str],
    filename: str,
) -> None:
    missing = required - set(data)

    if missing:
        raise RuntimeError(
            f"{filename}에 필수 항목이 없습니다: "
            + ", ".join(sorted(missing))
        )


def validate_configs(
    aug_config: dict,
    vae_config: dict,
) -> None:
    check_required_keys(
        aug_config,
        REQUIRED_AUG_KEYS,
        "day09_augmentation.json",
    )
    check_required_keys(
        vae_config,
        REQUIRED_VAE_KEYS,
        "day09_vae.json",
    )

    jitter_ratio = float(
        aug_config["jitter_ratio"]
    )
    compare_ratios = [
        float(value)
        for value in aug_config[
            "jitter_ratios_compare"
        ]
    ]

    if jitter_ratio < 0:
        raise RuntimeError(
            "jitter_ratio는 0 이상이어야 합니다."
        )

    if (
        not compare_ratios
        or any(
            value < 0
            for value in compare_ratios
        )
    ):
        raise RuntimeError(
            "jitter_ratios_compare에는 "
            "0 이상의 값이 1개 이상 필요합니다."
        )

    window_size = int(
        vae_config["window_valid_frames"]
    )
    stride = int(
        vae_config["window_stride"]
    )
    hidden_dim = int(
        vae_config["hidden_dim"]
    )
    latent_dim = int(
        vae_config["latent_dim"]
    )
    epochs = int(
        vae_config["epochs"]
    )
    batch_size = int(
        vae_config["batch_size"]
    )
    learning_rate = float(
        vae_config["learning_rate"]
    )
    beta = float(
        vae_config["beta"]
    )
    percentile = float(
        vae_config["threshold_percentile"]
    )
    compare_percentiles = [
        float(value)
        for value in vae_config[
            "thresholds_compare"
        ]
    ]

    if window_size < 2:
        raise RuntimeError(
            "window_valid_frames는 2 이상이어야 합니다."
        )

    if stride < 1:
        raise RuntimeError(
            "window_stride는 1 이상이어야 합니다."
        )

    if hidden_dim < 2:
        raise RuntimeError(
            "hidden_dim은 2 이상이어야 합니다."
        )

    if latent_dim < 1:
        raise RuntimeError(
            "latent_dim은 1 이상이어야 합니다."
        )

    if epochs < 1:
        raise RuntimeError(
            "epochs는 1 이상이어야 합니다."
        )

    if batch_size < 1:
        raise RuntimeError(
            "batch_size는 1 이상이어야 합니다."
        )

    if learning_rate <= 0:
        raise RuntimeError(
            "learning_rate는 0보다 커야 합니다."
        )

    if beta < 0:
        raise RuntimeError(
            "beta는 0 이상이어야 합니다."
        )

    if not 0 <= percentile <= 100:
        raise RuntimeError(
            "threshold_percentile은 "
            "0~100 범위여야 합니다."
        )

    if (
        not compare_percentiles
        or any(
            not 0 <= value <= 100
            for value in compare_percentiles
        )
    ):
        raise RuntimeError(
            "thresholds_compare에는 "
            "0~100 범위의 값이 1개 이상 필요합니다."
        )


def count_windows(
    valid_frames: int,
    window_size: int,
    stride: int,
) -> int:
    if valid_frames < window_size:
        return 0

    return (
        1
        + (
            valid_frames
            - window_size
        )
        // stride
    )


def main() -> None:
    required_files = [
        MODEL_PATH,
        FEATURE_PATH,
        TRAIN_INFO_PATH,
        AUG_CONFIG_PATH,
        VAE_CONFIG_PATH,
    ]

    for split_name in EXPECTED:
        required_files.append(
            SPLIT_DIR / f"{split_name}.csv"
        )
        required_files.append(
            SEQUENCE_DIR / f"{split_name}.csv"
        )

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)

    training_info = load_json(TRAIN_INFO_PATH)
    feature_columns = load_json(FEATURE_PATH)
    vae_config = load_json(VAE_CONFIG_PATH)
    aug_config = load_json(AUG_CONFIG_PATH)
    model = joblib.load(MODEL_PATH)

    check_required_keys(
        training_info,
        REQUIRED_TRAIN_INFO_KEYS,
        "rf_training_info.json",
    )
    validate_configs(
        aug_config,
        vae_config,
    )

    sequence_steps = int(
        training_info["sequence_steps"]
    )
    min_valid_frames = int(
        training_info["min_valid_frames"]
    )

    if sequence_steps < 2:
        raise RuntimeError(
            "sequence_steps는 2 이상이어야 합니다."
        )

    if min_valid_frames < 2:
        raise RuntimeError(
            "min_valid_frames는 2 이상이어야 합니다."
        )

    expected_feature_count = sequence_steps * 38
    expected_columns = sequence_feature_names(
        sequence_steps
    )

    if feature_columns != expected_columns:
        raise RuntimeError(
            "6일차 Feature Column 순서와 "
            "현재 sequence_utils.py가 다릅니다."
        )

    if len(feature_columns) != expected_feature_count:
        raise RuntimeError(
            "Feature Column 수가 "
            "sequence_steps × 38과 다릅니다."
        )

    if int(training_info["feature_count"]) != (
        expected_feature_count
    ):
        raise RuntimeError(
            "rf_training_info.json의 feature_count가 "
            "sequence_steps × 38과 다릅니다."
        )

    if int(model.n_features_in_) != expected_feature_count:
        raise RuntimeError(
            "Random Forest 입력 Feature 수가 다릅니다."
        )

    model_classes = set(map(str, model.classes_))
    info_classes = set(
        map(str, training_info["classes"])
    )

    if (
        model_classes != EXPECTED_ACTIONS
        or info_classes != EXPECTED_ACTIONS
    ):
        raise RuntimeError(
            "6일차 Action Class가 현재 6개 Action과 "
            "일치하지 않습니다."
        )

    normal_label = str(
        vae_config["normal_action_label"]
    )
    window_size = int(
        vae_config["window_valid_frames"]
    )
    stride = int(
        vae_config["window_stride"]
    )

    if normal_label not in EXPECTED_ACTIONS:
        raise RuntimeError(
            "VAE normal_action_label이 "
            "현재 Action에 없습니다."
        )

    split_frames = {}
    rows = []

    for split_name, (subject, count) in EXPECTED.items():
        split_path = SPLIT_DIR / f"{split_name}.csv"
        df = pd.read_csv(split_path)
        split_frames[split_name] = df

        if len(df) != count:
            raise RuntimeError(
                f"{split_name} Clip 수가 "
                f"{count}개가 아닙니다."
            )

        if set(df["subject"].astype(str)) != {subject}:
            raise RuntimeError(
                f"{split_name} Subject를 확인하세요."
            )

        if set(df["action_label"].astype(str)) != (
            EXPECTED_ACTIONS
        ):
            raise RuntimeError(
                f"{split_name}의 6개 Action Coverage를 "
                "확인하세요."
            )

        for _, item in df.iterrows():
            keypoint_path = ROOT / str(
                item["keypoint_csv"]
            )

            if not keypoint_path.exists():
                raise FileNotFoundError(keypoint_path)

            keypoint_df = pd.read_csv(keypoint_path)

            if keypoint_df.empty:
                raise RuntimeError(
                    "Keypoint CSV가 비어 있습니다: "
                    f"{item['clip_id']}"
                )

            frame_matrix, _ = build_frame_feature_matrix(
                keypoint_df
            )
            window_count = count_windows(
                len(frame_matrix),
                window_size,
                stride,
            )

            rows.append(
                {
                    "split": split_name,
                    "clip_id": item["clip_id"],
                    "subject": item["subject"],
                    "action_label": item["action_label"],
                    "valid_feature_frames": len(
                        frame_matrix
                    ),
                    "day06_min_valid_frames": (
                        min_valid_frames
                    ),
                    "vae_window_count": window_count,
                    "vae_window_ready": int(
                        window_count > 0
                    ),
                }
            )

    for split_name in EXPECTED:
        sequence_df = pd.read_csv(
            SEQUENCE_DIR / f"{split_name}.csv"
        )

        if len(sequence_df) != 6:
            raise RuntimeError(
                f"day06 {split_name}.csv는 "
                "6행이어야 합니다."
            )

        missing_columns = [
            column
            for column in feature_columns
            if column not in sequence_df.columns
        ]

        if missing_columns:
            raise RuntimeError(
                f"day06 {split_name}.csv의 "
                "Feature Column을 확인하세요."
            )

    val_not_ready = [
        row["clip_id"]
        for row in rows
        if (
            row["split"] == "val"
            and row["vae_window_count"] == 0
        )
    ]

    if val_not_ready:
        raise RuntimeError(
            "Validation에서 VAE Window를 만들 수 없는 "
            "Clip이 있습니다: "
            + ", ".join(val_not_ready)
        )

    train_normal = split_frames["train"][
        split_frames["train"][
            "action_label"
        ].astype(str)
        == normal_label
    ]
    val_normal = split_frames["val"][
        split_frames["val"][
            "action_label"
        ].astype(str)
        == normal_label
    ]

    if len(train_normal) != 1 or len(val_normal) != 1:
        raise RuntimeError(
            "현재 교육용 구조에서는 Train과 "
            "Validation에 Normal Reference가 "
            "각각 1개 필요합니다."
        )

    window_lookup = {
        (row["split"], row["clip_id"]): row[
            "vae_window_count"
        ]
        for row in rows
    }

    train_windows = window_lookup[
        (
            "train",
            train_normal.iloc[0]["clip_id"],
        )
    ]
    val_windows = window_lookup[
        (
            "val",
            val_normal.iloc[0]["clip_id"],
        )
    ]

    if train_windows < 3:
        raise RuntimeError(
            "S01 Normal Reference에서 "
            "VAE 학습 Window가 3개 미만입니다. "
            "window_valid_frames 또는 "
            "window_stride를 확인하세요."
        )

    if val_windows < 3:
        raise RuntimeError(
            "S02 Normal Reference에서 "
            "Threshold 계산 Window가 3개 미만입니다. "
            "window_valid_frames 또는 "
            "window_stride를 확인하세요."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Sequence steps : {sequence_steps}")
    print(f"Feature count  : {expected_feature_count}")
    print(f"Normal label   : {normal_label}")
    print(
        f"VAE window     : "
        f"{window_size} / stride {stride}"
    )
    print(
        f"Normal windows : "
        f"Train {train_windows} / Val {val_windows}"
    )
    print("Train / Val / Test: 6 / 6 / 6")
    print("Action classes     : OK")
    print("Keypoint CSV       : OK")
    print("Day06 artifacts    : OK")
    print()
    print(f"Saved: {OUTPUT_PATH}")
    print("Day09 input data: READY")


if __name__ == "__main__":
    main()