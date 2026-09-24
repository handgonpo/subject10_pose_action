from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
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

VAL_PATH = (
    ROOT
    / "data"
    / "day06"
    / "sequences"
    / "val.csv"
)

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "experiments"
    / "feature_set_comparison.csv"
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

ANGLE_NAMES = {
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
}


def load_config():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def feature_columns(
    df: pd.DataFrame,
):
    return [
        column
        for column in df.columns
        if column
        not in META_COLUMNS
    ]


def coordinate_only_columns(
    columns,
):
    selected = []

    for column in columns:
        is_angle = any(
            column.endswith(
                f"_{angle}"
            )
            for angle in (
                ANGLE_NAMES
            )
        )

        if not is_angle:
            selected.append(
                column
            )

    return selected


def evaluate(
    name,
    train,
    val,
    columns,
    config,
):
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

    X_val = (
        val[columns]
        .to_numpy(
            dtype=float
        )
    )

    y_val = (
        val[
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
            class_weight="balanced",
            n_jobs=-1,
        )
    )

    model.fit(
        X_train,
        y_train,
    )

    pred = model.predict(
        X_val
    )

    labels = sorted(
        set(y_train)
    )

    return {
        "experiment": name,
        "feature_count": (
            len(columns)
        ),
        "accuracy": (
            accuracy_score(
                y_val,
                pred,
            )
        ),
        "macro_f1": (
            f1_score(
                y_val,
                pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


def main() -> None:
    config = load_config()

    train = pd.read_csv(
        TRAIN_PATH
    )
    val = pd.read_csv(
        VAL_PATH
    )

    all_columns = (
        feature_columns(train)
    )

    coordinate_columns = (
        coordinate_only_columns(
            all_columns
        )
    )

    rows = [
        evaluate(
            "coordinates_plus_angles",
            train,
            val,
            all_columns,
            config,
        ),
        evaluate(
            "coordinates_only",
            train,
            val,
            coordinate_columns,
            config,
        ),
    ]

    result = pd.DataFrame(
        rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        result.to_string(
            index=False
        )
    )
    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()