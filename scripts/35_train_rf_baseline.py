from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
)


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT
    / "configs"
    / "day06_baseline.json"
)

TRAIN_PATH = (
    ROOT
    / "data"
    / "day06"
    / "sequences"
    / "train.csv"
)

MODEL_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_action_baseline.joblib"
)

FEATURE_LIST_PATH = (
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

META_COLUMNS = {
    "clip_id",
    "subject",
    "action_id",
    "action_label",
    "take",
    "valid_frames",
    "first_valid_frame",
    "last_valid_frame",
}


def load_config():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in META_COLUMNS
    ]


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            TRAIN_PATH
        )

    config = load_config()

    train = pd.read_csv(
        TRAIN_PATH
    )

    if len(train) != 6:
        raise RuntimeError(
            "현재 Baseline Train은 "
            "6개 Clip이어야 합니다."
        )

    classes = sorted(
        train[
            "action_label"
        ]
        .astype(str)
        .unique()
    )

    if len(classes) != 6:
        raise RuntimeError(
            "Train에 6개 Action Class가 "
            "모두 필요합니다."
        )

    columns = feature_columns(
        train
    )

    expected_features = (
        int(
            config[
                "sequence_steps"
            ]
        )
        * 38
    )

    if len(columns) != (
        expected_features
    ):
        raise RuntimeError(
            "Sequence Feature 수가 "
            f"{expected_features}개가 "
            f"아닙니다: {len(columns)}"
        )

    X_train = (
        train[columns]
        .to_numpy(
            dtype=float
        )
    )

    y_train = (
        train[
            "action_label"
        ]
        .astype(str)
        .to_numpy()
    )

    model = (
        RandomForestClassifier(
            n_estimators=int(
                config[
                    "n_estimators"
                ]
            ),
            max_depth=(
                config[
                    "max_depth"
                ]
            ),
            random_state=int(
                config[
                    "random_state"
                ]
            ),
            class_weight=(
                "balanced"
            ),
            n_jobs=-1,
        )
    )

    model.fit(
        X_train,
        y_train,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    with FEATURE_LIST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            columns,
            file,
            ensure_ascii=False,
            indent=2,
        )

    training_info = {
        "train_clips": len(train),
        "subjects": sorted(
            train[
                "subject"
            ]
            .astype(str)
            .unique()
            .tolist()
        ),
        "classes": classes,
        "feature_count": (
            len(columns)
        ),
        "sequence_steps": int(
            config[
                "sequence_steps"
            ]
        ),
        "min_valid_frames": int(
            config[
                "min_valid_frames"
            ]
        ),
        "n_estimators": int(
            config[
                "n_estimators"
            ]
        ),
        "max_depth": (
            config[
                "max_depth"
            ]
        ),
        "random_state": int(
            config[
                "random_state"
            ]
        ),
    }

    with TRAIN_INFO_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            training_info,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Train clips   : "
        f"{len(train)}"
    )
    print(
        f"Feature count : "
        f"{len(columns)}"
    )
    print(
        f"Classes       : "
        f"{classes}"
    )
    print()
    print(
        "Training completed."
    )
    print(
        f"Model: {MODEL_PATH}"
    )
    print(
        f"Features: "
        f"{FEATURE_LIST_PATH}"
    )


if __name__ == "__main__":
    main()