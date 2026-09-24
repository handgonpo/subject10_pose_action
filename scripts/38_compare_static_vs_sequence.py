from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(ROOT),
)


from src.sequence_utils import (
    build_frame_feature_matrix,
    middle_frame_feature,
)


CONFIG_PATH = (
    ROOT
    / "configs"
    / "day06_baseline.json"
)

TRAIN_SPLIT = (
    ROOT
    / "data"
    / "day05"
    / "splits"
    / "train.csv"
)

VAL_SPLIT = (
    ROOT
    / "data"
    / "day05"
    / "splits"
    / "val.csv"
)

SEQUENCE_TRAIN = (
    ROOT
    / "data"
    / "day06"
    / "sequences"
    / "train.csv"
)

SEQUENCE_VAL = (
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
    / "static_vs_sequence.csv"
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


def make_model(
    config: dict,
):
    return RandomForestClassifier(
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


def build_static_dataset(
    manifest_path: Path,
):
    manifest = pd.read_csv(
        manifest_path
    )

    X = []
    y = []

    for _, item in (
        manifest.iterrows()
    ):
        keypoint_path = (
            ROOT
            / str(
                item[
                    "keypoint_csv"
                ]
            )
        )

        df = pd.read_csv(
            keypoint_path
        )

        matrix, _ = (
            build_frame_feature_matrix(
                df
            )
        )

        if len(matrix) == 0:
            raise RuntimeError(
                "Static Feature를 "
                "만들 수 없습니다: "
                f"{item['clip_id']}"
            )

        feature = (
            middle_frame_feature(
                matrix
            )
        )

        X.append(feature)
        y.append(
            str(
                item[
                    "action_label"
                ]
            )
        )

    return (
        np.stack(X),
        np.asarray(y),
    )


def sequence_xy(
    path: Path,
):
    df = pd.read_csv(path)

    columns = [
        column
        for column in df.columns
        if column
        not in META_COLUMNS
    ]

    X = (
        df[columns]
        .to_numpy(
            dtype=float
        )
    )

    y = (
        df[
            "action_label"
        ]
        .astype(str)
        .to_numpy()
    )

    return X, y


def score_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
):
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

    return (
        accuracy_score(
            y_val,
            pred,
        ),
        f1_score(
            y_val,
            pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
    )


def main() -> None:
    config = load_config()

    (
        static_X_train,
        static_y_train,
    ) = build_static_dataset(
        TRAIN_SPLIT
    )

    (
        static_X_val,
        static_y_val,
    ) = build_static_dataset(
        VAL_SPLIT
    )

    static_acc, static_f1 = (
        score_model(
            make_model(config),
            static_X_train,
            static_y_train,
            static_X_val,
            static_y_val,
        )
    )

    (
        sequence_X_train,
        sequence_y_train,
    ) = sequence_xy(
        SEQUENCE_TRAIN
    )

    (
        sequence_X_val,
        sequence_y_val,
    ) = sequence_xy(
        SEQUENCE_VAL
    )

    sequence_acc, sequence_f1 = (
        score_model(
            make_model(config),
            sequence_X_train,
            sequence_y_train,
            sequence_X_val,
            sequence_y_val,
        )
    )

    result = pd.DataFrame(
        [
            {
                "representation": (
                    "middle_frame_static"
                ),
                "feature_count": 38,
                "accuracy": (
                    static_acc
                ),
                "macro_f1": (
                    static_f1
                ),
            },
            {
                "representation": (
                    f"{int(config['sequence_steps'])}"
                    "_step_sequence"
                ),
                "feature_count": (
                    sequence_X_train
                    .shape[1]
                ),
                "accuracy": (
                    sequence_acc
                ),
                "macro_f1": (
                    sequence_f1
                ),
            },
        ]
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