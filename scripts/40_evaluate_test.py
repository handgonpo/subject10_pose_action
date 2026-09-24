from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    ROOT
    / "data"
    / "day06"
    / "sequences"
    / "test.csv"
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

METRIC_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "metrics"
    / "test_metrics.txt"
)

PREDICTION_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "metrics"
    / "test_predictions.csv"
)

CONFUSION_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "confusion"
    / "test_confusion_matrix.png"
)


def load_feature_columns():
    with FEATURE_LIST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main() -> None:
    model = joblib.load(
        MODEL_PATH
    )

    test = pd.read_csv(
        TEST_PATH
    )

    columns = (
        load_feature_columns()
    )

    missing = [
        column
        for column in columns
        if column not in test.columns
    ]

    if missing:
        raise RuntimeError(
            "Test에 학습 Feature가 "
            "누락되었습니다."
        )

    X_test = (
        test[columns]
        .to_numpy(
            dtype=float
        )
    )

    y_true = (
        test[
            "action_label"
        ]
        .astype(str)
        .to_numpy()
    )

    y_pred = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(
            X_test
        )
    )

    confidence = (
        probabilities.max(
            axis=1
        )
    )

    labels = list(
        model.classes_
    )

    accuracy = (
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    macro_f1 = (
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )

    report = (
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )

    print(
        f"Test Accuracy: "
        f"{accuracy:.4f}"
    )
    print(
        f"Test Macro F1: "
        f"{macro_f1:.4f}"
    )
    print()
    print(report)

    METRIC_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRIC_PATH.write_text(
        (
            f"Test Accuracy: "
            f"{accuracy:.6f}\n"
            f"Test Macro F1: "
            f"{macro_f1:.6f}\n\n"
            f"{report}"
        ),
        encoding="utf-8",
    )

    prediction_df = (
        pd.DataFrame(
            {
                "clip_id": (
                    test["clip_id"]
                ),
                "subject": (
                    test["subject"]
                ),
                "true_label": (
                    y_true
                ),
                "pred_label": (
                    y_pred
                ),
                "confidence": (
                    confidence
                ),
                "correct": (
                    y_true == y_pred
                ),
            }
        )
    )

    prediction_df.to_csv(
        PREDICTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    display = (
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=labels,
        )
    )

    display.plot(
        xticks_rotation=45,
    )

    plt.title(
        "Test Confusion Matrix"
    )
    plt.tight_layout()

    CONFUSION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        CONFUSION_PATH,
        dpi=150,
    )
    plt.close()

    print(
        f"Metrics: "
        f"{METRIC_PATH}"
    )
    print(
        f"Predictions: "
        f"{PREDICTION_PATH}"
    )
    print(
        f"Confusion Matrix: "
        f"{CONFUSION_PATH}"
    )


if __name__ == "__main__":
    main()