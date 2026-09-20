from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help=(
            "예: reports/day04/"
            "sequences/wave_sequence.csv"
        ),
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=5,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = Path(
        args.source
    )

    if not source.exists():
        raise FileNotFoundError(
            source
        )

    if args.samples < 2:
        raise ValueError(
            "--samples는 "
            "2 이상이어야 합니다."
        )

    df = pd.read_csv(source)

    required_columns = {
        "action_label",
        "frame_id",
        "time_sec",
        "feature_ok",
        "hip_center_x",
        "hip_center_y",
        "left_wrist_x",
        "left_wrist_y",
        "right_wrist_x",
        "right_wrist_y",
        "left_ankle_x",
        "left_ankle_y",
        "right_ankle_x",
        "right_ankle_y",
        "left_elbow_angle",
        "right_elbow_angle",
        "left_knee_angle",
        "right_knee_angle",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Sequence CSV에 "
            "필수 열이 없습니다: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    valid = df[
        df["feature_ok"] == 1
    ].copy()

    if len(valid) < 2:
        raise RuntimeError(
            "비교할 수 있는 정상 "
            "Feature Row가 2개 미만입니다."
        )

    valid = valid.sort_values(
        "frame_id"
    ).reset_index(
        drop=True
    )

    sample_count = min(
        args.samples,
        len(valid),
    )

    indices = np.linspace(
        0,
        len(valid) - 1,
        sample_count,
    ).round().astype(int)

    selected = valid.iloc[
        indices
    ].copy()

    columns = [
        "frame_id",
        "time_sec",
        "hip_center_x",
        "hip_center_y",
        "left_wrist_x",
        "left_wrist_y",
        "right_wrist_x",
        "right_wrist_y",
        "left_ankle_x",
        "left_ankle_y",
        "right_ankle_x",
        "right_ankle_y",
        "left_elbow_angle",
        "right_elbow_angle",
        "left_knee_angle",
        "right_knee_angle",
    ]

    print(
        f"Source: {source.name}"
    )

    print(
        f"Action: "
        f"{valid['action_label'].iloc[0]}"
    )

    print(
        f"Valid feature rows: "
        f"{len(valid)}"
    )

    print(
        f"Selected samples: "
        f"{sample_count}"
    )

    print()

    print(
        selected[
            columns
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()