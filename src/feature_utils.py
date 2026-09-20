from __future__ import annotations

import math

import numpy as np


# COCO 17 index
NOSE = 0

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

LEFT_ELBOW = 7
RIGHT_ELBOW = 8

LEFT_WRIST = 9
RIGHT_WRIST = 10

LEFT_HIP = 11
RIGHT_HIP = 12

LEFT_KNEE = 13
RIGHT_KNEE = 14

LEFT_ANKLE = 15
RIGHT_ANKLE = 16


LEFT_RIGHT_PAIRS = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
]


def calculate_angle(
    point_a,
    point_b,
    point_c,
) -> float:
    """
    point_b를 중심으로
    A-B-C 세 점의 각도를 계산합니다.
    """

    a = np.asarray(
        point_a,
        dtype=np.float32,
    )

    b = np.asarray(
        point_b,
        dtype=np.float32,
    )

    c = np.asarray(
        point_c,
        dtype=np.float32,
    )

    vector_ba = a - b
    vector_bc = c - b

    norm_ba = np.linalg.norm(
        vector_ba
    )

    norm_bc = np.linalg.norm(
        vector_bc
    )

    if (
        norm_ba < 1e-8
        or norm_bc < 1e-8
    ):
        return float("nan")

    cosine = np.dot(
        vector_ba,
        vector_bc,
    ) / (
        norm_ba
        * norm_bc
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    angle = math.degrees(
        math.acos(
            float(cosine)
        )
    )

    return float(angle)


def midpoint(
    point_a,
    point_b,
):
    a = np.asarray(
        point_a,
        dtype=np.float32,
    )

    b = np.asarray(
        point_b,
        dtype=np.float32,
    )

    return (
        a + b
    ) / 2.0


def body_center(
    keypoints,
):
    """
    좌·우 골반의 중점을
    몸 중심으로 사용합니다.
    """

    return midpoint(
        keypoints[LEFT_HIP],
        keypoints[RIGHT_HIP],
    )


def shoulder_width(
    keypoints,
) -> float:
    """
    좌·우 어깨 사이 거리를 계산합니다.
    """

    left = np.asarray(
        keypoints[LEFT_SHOULDER],
        dtype=np.float32,
    )

    right = np.asarray(
        keypoints[RIGHT_SHOULDER],
        dtype=np.float32,
    )

    return float(
        np.linalg.norm(
            left - right
        )
    )


def center_keypoints(
    keypoints,
):
    """
    골반 중심이 (0, 0)이 되도록
    모든 관절 좌표를 이동합니다.
    """

    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    center = body_center(
        keypoints
    )

    centered = (
        keypoints
        - center
    )

    return centered, center


def normalize_by_shoulder_width(
    keypoints,
):
    """
    몸 중심 이동 후
    어깨 폭을 기준으로 크기를 정규화합니다.
    """

    centered, center = (
        center_keypoints(
            keypoints
        )
    )

    scale = shoulder_width(
        keypoints
    )

    if scale < 1e-8:
        raise ValueError(
            "어깨 폭을 계산할 수 없습니다."
        )

    normalized = (
        centered
        / scale
    )

    return (
        normalized,
        center,
        scale,
    )


def flip_keypoints_horizontal(
    normalized_keypoints,
):
    """
    몸 중심 기준 정규화된 좌표를
    좌우 반전합니다.

    1. x 좌표의 부호를 반전
    2. Left/Right 관절 의미도 교환
    """

    flipped = np.asarray(
        normalized_keypoints,
        dtype=np.float32,
    ).copy()

    flipped[:, 0] *= -1.0

    for left_id, right_id in (
        LEFT_RIGHT_PAIRS
    ):
        temp = flipped[
            left_id
        ].copy()

        flipped[left_id] = (
            flipped[right_id]
        )

        flipped[right_id] = (
            temp
        )

    return flipped


def selected_joint_angles(
    keypoints,
):
    """
    자주 사용하는 관절각도를 계산합니다.
    """

    return {
        "left_elbow_angle": (
            calculate_angle(
                keypoints[
                    LEFT_SHOULDER
                ],
                keypoints[
                    LEFT_ELBOW
                ],
                keypoints[
                    LEFT_WRIST
                ],
            )
        ),

        "right_elbow_angle": (
            calculate_angle(
                keypoints[
                    RIGHT_SHOULDER
                ],
                keypoints[
                    RIGHT_ELBOW
                ],
                keypoints[
                    RIGHT_WRIST
                ],
            )
        ),

        "left_knee_angle": (
            calculate_angle(
                keypoints[
                    LEFT_HIP
                ],
                keypoints[
                    LEFT_KNEE
                ],
                keypoints[
                    LEFT_ANKLE
                ],
            )
        ),

        "right_knee_angle": (
            calculate_angle(
                keypoints[
                    RIGHT_HIP
                ],
                keypoints[
                    RIGHT_KNEE
                ],
                keypoints[
                    RIGHT_ANKLE
                ],
            )
        ),
    }