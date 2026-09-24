from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    ROOT
    / "data"
    / "day09"
    / "sequences_augmented"
    / "train.csv"
)
DAY06_FEATURE_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_feature_columns.json"
)
DAY06_INFO_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_training_info.json"
)
MODEL_PATH = (
    ROOT
    / "artifacts"
    / "day09"
    / "augmentation"
    / "rf_action_augmented.joblib"
)
DEFAULT_INFO_PATH = (
    ROOT
    / "artifacts"
    / "day09"
    / "augmentation"
    / "rf_augmented_training_info.json"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        default=str(
            TRAIN_PATH
        ),
    )

    parser.add_argument(
        "--model-output",
        default=str(
            MODEL_PATH
        ),
    )

    parser.add_argument(
        "--info-output",
        default=str(
            DEFAULT_INFO_PATH
        ),
    )

    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    args = parse_args()

    train_path = Path(
        args.train
    )
    model_path = Path(
        args.model_output
    )
    info_path = Path(
        args.info_output
    )

    if not train_path.is_absolute():
        train_path = ROOT / train_path

    if not model_path.is_absolute():
        model_path = ROOT / model_path

    if not info_path.is_absolute():
        info_path = ROOT / info_path

    train = pd.read_csv(train_path)
    feature_columns = load_json(DAY06_FEATURE_PATH)
    day06_info = load_json(DAY06_INFO_PATH)

    missing = [
        column
        for column in feature_columns
        if column not in train.columns
    ]

    if missing:
        raise RuntimeError(
            "Augmented Train에 6일차 Feature가 없습니다."
        )

    X_train = train[feature_columns].to_numpy(
        dtype=float
    )
    y_train = train["action_label"].astype(str).to_numpy()

    if len(set(y_train)) != 6:
        raise RuntimeError(
            "Train에 6개 Action Class가 모두 필요합니다."
        )

    model = RandomForestClassifier(
        n_estimators=int(day06_info["n_estimators"]),
        max_depth=day06_info["max_depth"],
        random_state=int(day06_info["random_state"]),
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    joblib.dump(model, model_path)

    training_info = {
        "train_rows": len(train),
        "source_train_clips": int(
            train["source_clip_id"].nunique()
        ),
        "classes": sorted(set(y_train)),
        "feature_count": len(feature_columns),
        "sequence_steps": int(
            day06_info["sequence_steps"]
        ),
        "n_estimators": int(
            day06_info["n_estimators"]
        ),
        "max_depth": day06_info["max_depth"],
        "random_state": int(
            day06_info["random_state"]
        ),
    }

    info_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    info_path.write_text(
        json.dumps(
            training_info,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Train rows    : {len(train)}")
    print(f"Feature count : {len(feature_columns)}")
    print(f"Classes       : {sorted(set(y_train))}")
    print(f"Train         : {train_path}")
    print(f"Model         : {model_path}")
    print(f"Info          : {info_path}")


if __name__ == "__main__":
    main()