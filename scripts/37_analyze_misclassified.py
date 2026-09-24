from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PREDICTION_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "metrics"
    / "validation_predictions.csv"
)

VAL_SPLIT_PATH = (
    ROOT
    / "data"
    / "day05"
    / "splits"
    / "val.csv"
)

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "misclassified"
    / "validation_misclassified.csv"
)


def main() -> None:
    predictions = pd.read_csv(
        PREDICTION_PATH
    )

    val_manifest = pd.read_csv(
        VAL_SPLIT_PATH
    )

    merged = predictions.merge(
        val_manifest[
            [
                "clip_id",
                "video_path",
                "keypoint_csv",
                "action_label",
            ]
        ],
        on="clip_id",
        how="left",
        suffixes=(
            "",
            "_manifest",
        ),
    )

    wrong = merged[
        merged[
            "true_label"
        ]
        != merged[
            "pred_label"
        ]
    ].copy()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    wrong.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Validation clips : "
        f"{len(predictions)}"
    )
    print(
        f"Misclassified    : "
        f"{len(wrong)}"
    )

    if wrong.empty:
        print()
        print(
            "Validation 오분류가 "
            "없습니다."
        )
    else:
        print()
        print(
            wrong[
                [
                    "clip_id",
                    "true_label",
                    "pred_label",
                    "confidence",
                    "video_path",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()