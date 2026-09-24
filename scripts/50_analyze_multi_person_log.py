from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR = (
    ROOT / "reports" / "day08" / "ip_webcam"
)

DEFAULT_OUTPUT_PATH = (
    DEFAULT_INPUT_DIR
    / "ip_multi_runtime_summary.csv"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log",
        default=None,
        help=(
            "생략하면 입력 폴더의 가장 최근 "
            "ip_multi_*.csv 사용"
        ),
    )

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


def choose_log(value, input_dir: Path, output_path: Path) -> Path:
    if value:
        path = Path(value)

        if not path.is_absolute():
            path = ROOT / path

        return path

    files = sorted(input_dir.glob("ip_multi_*.csv"))

    files = [
        path
        for path in files
        if path.resolve()
        != output_path.resolve()
    ]

    if not files:
        raise RuntimeError(
            "Multi-Person IP Webcam Log가 없습니다."
        )

    return files[-1]


def most_common_stable_label(
    group: pd.DataFrame,
) -> str:
    labels = (
        group["stable_label"]
        .fillna("")
        .astype(str)
    )

    labels = labels[
        ~labels.isin(
            ["", "WARMING_UP", "UNKNOWN"]
        )
    ]

    if labels.empty:
        return ""

    return str(labels.value_counts().index[0])


def main() -> None:
    args = parse_args()

    input_dir = Path(
        args.input_dir
    )
    output_path = Path(
        args.output
    )

    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir

    if not output_path.is_absolute():
        output_path = ROOT / output_path

    log_path = choose_log(
        args.log,
        input_dir,
        output_path,
    )

    if not log_path.exists():
        raise FileNotFoundError(log_path)

    df = pd.read_csv(log_path)

    if df.empty:
        raise RuntimeError("Runtime Log가 비어 있습니다.")

    rows = []

    for track_id, group in df.groupby("track_id"):
        updates = group[
            group["prediction_updated"] == 1
        ].copy()

        if updates.empty:
            unknown_ratio = ""
            mean_confidence = ""
            common_label = ""
        else:
            unknown_ratio = round(
                float(
                    (
                        updates["threshold_label"]
                        == "UNKNOWN"
                    ).mean()
                ),
                4,
            )

            mean_confidence = round(
                float(updates["confidence"].mean()),
                4,
            )

            common_label = most_common_stable_label(
                updates
            )

        fps_values = group.loc[
            group["processing_fps"] > 0,
            "processing_fps",
        ]

        rows.append(
            {
                "track_id": int(track_id),
                "first_seen_sec": round(
                    float(group["elapsed_sec"].min()),
                    3,
                ),
                "last_seen_sec": round(
                    float(group["elapsed_sec"].max()),
                    3,
                ),
                "observed_frames": int(
                    group["frame_id"].nunique()
                ),
                "prediction_updates": len(updates),
                "unknown_ratio": unknown_ratio,
                "most_common_stable_label": common_label,
                "mean_confidence": mean_confidence,
                "mean_processing_fps": (
                    round(float(fps_values.mean()), 3)
                    if not fps_values.empty
                    else 0.0
                ),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        "track_id"
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

    print(f"Log: {log_path}")
    print()
    print(result.to_string(index=False))
    print()
    print(
        "주의: 이 표는 Runtime 관찰용이며 "
        "정식 Action Accuracy가 아닙니다."
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()