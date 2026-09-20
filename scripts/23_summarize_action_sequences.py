from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
)


DEFAULT_SEQUENCE_DIR = (
    ROOT
    / "reports"
    / "day04"
    / "sequences"
)

DEFAULT_OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day04"
    / "tables"
    / "sequence_summary.csv"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-dir",
        default=str(
            DEFAULT_SEQUENCE_DIR
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
    )

    return parser.parse_args()


def coordinate_columns() -> list[str]:
    columns = []

    for name in COCO_KEYPOINT_NAMES:
        columns.append(
            f"{name}_x"
        )

        columns.append(
            f"{name}_y"
        )

    return columns


def mean_adjacent_feature_change(
    df: pd.DataFrame,
    columns: list[str],
) -> float:
    if len(df) < 2:
        return 0.0

    numeric = (
        df[columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    feature_ok = (
        df["feature_ok"]
        == 1
    )

    pair_ok = (
        feature_ok
        & feature_ok.shift(
            1,
            fill_value=False,
        )
    )

    differences = (
        numeric.diff()
    )

    valid_differences = (
        differences.loc[
            pair_ok
        ]
        .dropna()
    )

    if valid_differences.empty:
        return 0.0

    distances = np.linalg.norm(
        valid_differences.to_numpy(
            dtype=float
        ),
        axis=1,
    )

    return float(
        np.mean(distances)
    )


def main() -> None:
    args = parse_args()

    sequence_dir = Path(
        args.sequence_dir
    )

    output_path = Path(
        args.output
    )

    if not sequence_dir.exists():
        raise FileNotFoundError(
            sequence_dir
        )

    sequence_files = sorted(
        sequence_dir.glob(
            "*_sequence.csv"
        )
    )

    if not sequence_files:
        raise RuntimeError(
            "Sequence CSV가 없습니다."
        )

    coord_cols = (
        coordinate_columns()
    )

    rows = []

    required_columns = {
        "action_id",
        "action_label",
        "time_sec",
        "pose_ok",
        "feature_ok",
        "mean_keypoint_conf",
        "low_keypoint_count",
        *coord_cols,
    }

    for path in sequence_files:
        df = pd.read_csv(path)

        if df.empty:
            print(
                f"[SKIP] 빈 CSV: "
                f"{path.name}"
            )
            continue

        missing_columns = (
            required_columns
            - set(df.columns)
        )

        if missing_columns:
            raise RuntimeError(
                f"{path.name}에 "
                "필수 열이 없습니다: "
                + ", ".join(
                    sorted(
                        missing_columns
                    )
                )
            )

        valid_pose = df[
            df["pose_ok"] == 1
        ].copy()

        valid_feature = df[
            df["feature_ok"] == 1
        ].copy()

        total_rows = len(df)
        pose_rows = len(
            valid_pose
        )
        feature_rows = len(
            valid_feature
        )

        pose_ratio = (
            pose_rows / total_rows
            if total_rows > 0
            else 0.0
        )

        feature_ratio = (
            feature_rows / total_rows
            if total_rows > 0
            else 0.0
        )

        mean_conf = (
            float(
                valid_pose[
                    "mean_keypoint_conf"
                ].mean()
            )
            if pose_rows > 0
            else 0.0
        )

        mean_low = (
            float(
                valid_pose[
                    "low_keypoint_count"
                ].mean()
            )
            if pose_rows > 0
            else 17.0
        )

        movement = (
            mean_adjacent_feature_change(
                df,
                coord_cols,
            )
        )

        last_time = float(
            df["time_sec"].max()
        )

        rows.append(
            {
                "action_id": (
                    df[
                        "action_id"
                    ].iloc[0]
                ),
                "action_label": (
                    df[
                        "action_label"
                    ].iloc[0]
                ),
                "rows": total_rows,
                "pose_ok_rows": pose_rows,
                "pose_ok_ratio": round(
                    pose_ratio,
                    4,
                ),
                "feature_ok_rows":
                    feature_rows,
                "feature_ok_ratio":
                    round(
                        feature_ratio,
                        4,
                    ),
                "mean_keypoint_conf":
                    round(
                        mean_conf,
                        4,
                    ),
                "mean_low_keypoint_count":
                    round(
                        mean_low,
                        3,
                    ),
                "mean_adjacent_feature_change":
                    round(
                        movement,
                        6,
                    ),
                "last_time_sec": round(
                    last_time,
                    3,
                ),
            }
        )

    if not rows:
        raise RuntimeError(
            "요약할 수 있는 "
            "Sequence 데이터가 없습니다."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Saved: {output_path}"
    )

    print()

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()