from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]

VAL_PATH = (
    ROOT / "data" / "day06" / "sequences" / "val.csv"
)
FEATURE_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_feature_columns.json"
)
BASELINE_MODEL_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_action_baseline.joblib"
)
AUGMENTED_MODEL_PATH = (
    ROOT
    / "artifacts"
    / "day09"
    / "augmentation"
    / "rf_action_augmented.joblib"
)
METRICS_PATH = (
    ROOT
    / "reports"
    / "day09"
    / "augmentation"
    / "baseline_vs_augmented.csv"
)
PREDICTIONS_PATH = (
    ROOT
    / "reports"
    / "day09"
    / "augmentation"
    / "baseline_vs_augmented_predictions.csv"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--augmented-model",
        default=str(
            AUGMENTED_MODEL_PATH
        ),
    )

    parser.add_argument(
        "--metrics-output",
        default=str(
            METRICS_PATH
        ),
    )

    parser.add_argument(
        "--predictions-output",
        default=str(
            PREDICTIONS_PATH
        ),
    )

    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate(name, model, X, y_true):
    pred = model.predict(X)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, pred),
        "macro_f1": f1_score(
            y_true,
            pred,
            average="macro",
            zero_division=0,
        ),
    }

    return metrics, pred


def main() -> None:
    args = parse_args()

    augmented_model_path = Path(
        args.augmented_model
    )
    metrics_path = Path(
        args.metrics_output
    )
    predictions_path = Path(
        args.predictions_output
    )

    if not augmented_model_path.is_absolute():
        augmented_model_path = (
            ROOT / augmented_model_path
        )

    if not metrics_path.is_absolute():
        metrics_path = ROOT / metrics_path

    if not predictions_path.is_absolute():
        predictions_path = (
            ROOT / predictions_path
        )

    val = pd.read_csv(VAL_PATH)
    feature_columns = load_json(FEATURE_PATH)

    if len(val) != 6:
        raise RuntimeError(
            "현재 Validation은 S02의 6개 Clip이어야 합니다."
        )

    X_val = val[feature_columns].to_numpy(dtype=float)
    y_true = val["action_label"].astype(str).to_numpy()

    baseline = joblib.load(BASELINE_MODEL_PATH)
    augmented = joblib.load(
        augmented_model_path
    )

    expected_features = len(feature_columns)

    for model in [baseline, augmented]:
        if int(model.n_features_in_) != expected_features:
            raise RuntimeError(
                "모델 입력 Feature 수가 Validation과 다릅니다."
            )

    baseline_metrics, baseline_pred = evaluate(
        "baseline",
        baseline,
        X_val,
        y_true,
    )
    augmented_metrics, augmented_pred = evaluate(
        "augmented",
        augmented,
        X_val,
        y_true,
    )

    metrics = pd.DataFrame(
        [baseline_metrics, augmented_metrics]
    )

    predictions = val[
        [
            "clip_id",
            "subject",
            "action_id",
            "action_label",
        ]
    ].copy()
    predictions["baseline_pred"] = baseline_pred
    predictions["augmented_pred"] = augmented_pred
    predictions["baseline_correct"] = (
        predictions["action_label"]
        == predictions["baseline_pred"]
    )
    predictions["augmented_correct"] = (
        predictions["action_label"]
        == predictions["augmented_pred"]
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    predictions_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(metrics.to_string(index=False))
    print()
    print(predictions.to_string(index=False))
    print()
    print(f"Saved: {metrics_path}")
    print(f"Saved: {predictions_path}")


if __name__ == "__main__":
    main()