from __future__ import annotations

import numpy as np
import pandas as pd

from src.pose_utils import COCO_KEYPOINT_NAMES


LEFT_RIGHT_PAIRS = [
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]


def _xy_columns(name: str):
    return (
        f"{name}_x_px",
        f"{name}_y_px",
    )


def _copy_with_float_coordinates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Jitter처럼 실수 값을 대입할 수 있도록
    Pixel 좌표 열을 float dtype으로 복사합니다.
    """

    output = df.copy()

    for name in COCO_KEYPOINT_NAMES:
        x_col, y_col = _xy_columns(name)

        for column in [x_col, y_col]:
            if column in output.columns:
                output[column] = pd.to_numeric(
                    output[column],
                    errors="coerce",
                ).astype(float)

    return output


def estimate_hip_center_x(row: pd.Series):
    left_x = row.get("left_hip_x_px")
    right_x = row.get("right_hip_x_px")

    if pd.isna(left_x) or pd.isna(right_x):
        return None

    return (
        float(left_x) + float(right_x)
    ) / 2.0


def estimate_shoulder_width(row: pd.Series):
    lx = row.get("left_shoulder_x_px")
    ly = row.get("left_shoulder_y_px")
    rx = row.get("right_shoulder_x_px")
    ry = row.get("right_shoulder_y_px")

    values = [lx, ly, rx, ry]

    if any(pd.isna(value) for value in values):
        return None

    dx = float(lx) - float(rx)
    dy = float(ly) - float(ry)
    width = (dx * dx + dy * dy) ** 0.5

    if width <= 1e-6:
        return None

    return float(width)


def horizontal_flip_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    output = _copy_with_float_coordinates(df)

    for row_index in output.index:
        row = output.loc[row_index]
        center_x = estimate_hip_center_x(row)

        if center_x is None:
            continue

        for name in COCO_KEYPOINT_NAMES:
            x_col, _ = _xy_columns(name)
            value = output.at[row_index, x_col]

            if pd.isna(value):
                continue

            output.at[row_index, x_col] = (
                2.0 * center_x - float(value)
            )

        for left_name, right_name in LEFT_RIGHT_PAIRS:
            left_x, left_y = _xy_columns(left_name)
            right_x, right_y = _xy_columns(right_name)

            left_values = (
                output.at[row_index, left_x],
                output.at[row_index, left_y],
            )
            right_values = (
                output.at[row_index, right_x],
                output.at[row_index, right_y],
            )

            output.at[row_index, left_x] = right_values[0]
            output.at[row_index, left_y] = right_values[1]
            output.at[row_index, right_x] = left_values[0]
            output.at[row_index, right_y] = left_values[1]

            left_conf = f"{left_name}_conf"
            right_conf = f"{right_name}_conf"

            if (
                left_conf in output.columns
                and right_conf in output.columns
            ):
                left_conf_value = output.at[
                    row_index,
                    left_conf,
                ]
                right_conf_value = output.at[
                    row_index,
                    right_conf,
                ]

                output.at[
                    row_index,
                    left_conf,
                ] = right_conf_value
                output.at[
                    row_index,
                    right_conf,
                ] = left_conf_value

    return output


def jitter_dataframe(
    df: pd.DataFrame,
    jitter_ratio: float,
    random_state: int,
) -> pd.DataFrame:
    if float(jitter_ratio) < 0:
        raise ValueError(
            "jitter_ratio는 0 이상이어야 합니다."
        )

    rng = np.random.default_rng(random_state)
    output = _copy_with_float_coordinates(df)

    for row_index in output.index:
        row = output.loc[row_index]
        shoulder_width = estimate_shoulder_width(row)

        if shoulder_width is None:
            continue

        sigma = shoulder_width * float(jitter_ratio)

        for name in COCO_KEYPOINT_NAMES:
            x_col, y_col = _xy_columns(name)
            x = output.at[row_index, x_col]
            y = output.at[row_index, y_col]

            if pd.isna(x) or pd.isna(y):
                continue

            noise = rng.normal(
                loc=0.0,
                scale=sigma,
                size=2,
            )

            output.at[row_index, x_col] = (
                float(x) + float(noise[0])
            )
            output.at[row_index, y_col] = (
                float(y) + float(noise[1])
            )

    return output