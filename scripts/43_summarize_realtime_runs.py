from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR = (
    ROOT
    / "reports"
    / "day07"
    / "saved_video"
)

DEFAULT_OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day07"
    / "realtime_run_summary.csv"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default=str(
            DEFAULT_INPUT_DIR
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
    )

    return parser.parse_args()


def label_changes(
    series: pd.Series,
) -> int:
    clean = (
        series
        .fillna("")
        .astype(str)
    )

    if clean.empty:
        return 0

    return int(
        (
            clean
            != clean.shift(1)
        ).sum()
        - 1
    )


def main() -> None:
    args = parse_args()

    input_dir = Path(
        args.input_dir
    )

    output_path = Path(
        args.output
    )

    if not input_dir.is_absolute():
        input_dir = (
            ROOT
            / input_dir
        )

    if not output_path.is_absolute():
        output_path = (
            ROOT
            / output_path
        )

    files = sorted(
        input_dir.glob(
            "*.csv"
        )
    )

    if not files:
        raise RuntimeError(
            "Day07 저장영상 실행 CSV가 "
            "없습니다."
        )

    rows = []

    for path in files:
        df = pd.read_csv(
            path
        )

        if (
            "prediction_updated"
            not in df.columns
        ):
            continue

        prediction_rows = df[
            df[
                "prediction_updated"
            ]
            == 1
        ].copy()

        if prediction_rows.empty:
            continue

        expected_label = str(
            prediction_rows[
                "expected_label"
            ]
            .fillna("")
            .iloc[0]
        )

        unknown_ratio = float(
            (
                prediction_rows[
                    "threshold_label"
                ]
                == "UNKNOWN"
            ).mean()
        )

        raw_changes = (
            label_changes(
                prediction_rows[
                    "raw_label"
                ]
            )
        )

        stable_changes = (
            label_changes(
                prediction_rows[
                    "stable_label"
                ]
            )
        )

        if expected_label:
            stable_match_ratio = float(
                (
                    prediction_rows[
                        "stable_label"
                    ]
                    == expected_label
                ).mean()
            )
        else:
            stable_match_ratio = float(
                "nan"
            )

        fps_values = df.loc[
            df[
                "processing_fps"
            ]
            > 0,
            "processing_fps",
        ]

        rows.append(
            {
                "run": path.stem,
                "expected_label": (
                    expected_label
                ),
                "window_seconds": (
                    prediction_rows[
                        "window_seconds"
                    ].iloc[0]
                    if "window_seconds"
                    in prediction_rows.columns
                    else ""
                ),
                "confidence_threshold": (
                    prediction_rows[
                        "confidence_threshold"
                    ].iloc[0]
                    if "confidence_threshold"
                    in prediction_rows.columns
                    else ""
                ),
                "stabilization_buffer": (
                    prediction_rows[
                        "stabilization_buffer"
                    ].iloc[0]
                    if "stabilization_buffer"
                    in prediction_rows.columns
                    else ""
                ),
                "stabilization_min_votes": (
                    prediction_rows[
                        "stabilization_min_votes"
                    ].iloc[0]
                    if "stabilization_min_votes"
                    in prediction_rows.columns
                    else ""
                ),
                "prediction_updates": (
                    len(
                        prediction_rows
                    )
                ),
                "unknown_ratio": round(
                    unknown_ratio,
                    4,
                ),
                "raw_label_changes": (
                    raw_changes
                ),
                "stable_label_changes": (
                    stable_changes
                ),
                "stable_match_ratio": (
                    round(
                        stable_match_ratio,
                        4,
                    )
                    if pd.notna(
                        stable_match_ratio
                    )
                    else ""
                ),
                "mean_confidence": round(
                    float(
                        prediction_rows[
                            "confidence"
                        ].mean()
                    ),
                    4,
                ),
                "mean_processing_fps": round(
                    float(
                        fps_values.mean()
                    ),
                    3,
                )
                if not fps_values.empty
                else 0.0,
            }
        )

    if not rows:
        raise RuntimeError(
            "요약할 Prediction 결과가 "
            "없습니다."
        )

    result = pd.DataFrame(
        rows
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
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
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()