from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from src.common import (
    load_config,
    resolve_device,
)

from src.feature_utils import (
    flip_keypoints_horizontal,
    normalize_by_shoulder_width,
)

from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--left",
        default=str(
            ROOT
            / "data"
            / "day03"
            / "images"
            / "left_arm_up.jpg"
        ),
    )

    parser.add_argument(
        "--right",
        default=str(
            ROOT
            / "data"
            / "day03"
            / "images"
            / "right_arm_up.jpg"
        ),
    )

    return parser.parse_args()


def extract_normalized_pose(
    model,
    source: Path,
    config,
    device,
):
    results = model.predict(
        source=str(source),
        imgsz=config["image_size"],
        conf=config[
            "person_confidence"
        ],
        device=device,
        verbose=False,
    )

    result = results[0]

    if (
        result.keypoints is None
        or result.keypoints.xy is None
        or len(
            result.keypoints.xy
        ) == 0
    ):
        raise RuntimeError(
            f"Pose를 찾지 못했습니다: "
            f"{source.name}"
        )

    keypoints = (
        result.keypoints.xy[0]
        .cpu()
        .numpy()
    )

    normalized, _, _ = (
        normalize_by_shoulder_width(
            keypoints
        )
    )

    return normalized


def mean_joint_distance(
    pose_a,
    pose_b,
) -> float:
    distances = np.linalg.norm(
        pose_a - pose_b,
        axis=1,
    )

    return float(
        np.mean(distances)
    )


def main() -> None:
    args = parse_args()

    left_source = Path(
        args.left
    )

    right_source = Path(
        args.right
    )

    if not left_source.exists():
        raise FileNotFoundError(
            left_source
        )

    if not right_source.exists():
        raise FileNotFoundError(
            right_source
        )

    config = load_config()

    device = resolve_device(
        config["device"]
    )

    model = YOLO(
        config["pose_model"]
    )

    # 1. 실제 Left Arm Up 정규화
    left_normalized = (
        extract_normalized_pose(
            model,
            left_source,
            config,
            device,
        )
    )

    # 2. 실제 Right Arm Up 정규화
    right_normalized = (
        extract_normalized_pose(
            model,
            right_source,
            config,
            device,
        )
    )

    # 3. Left Pose를 좌우 반전
    flipped_left = (
        flip_keypoints_horizontal(
            left_normalized
        )
    )

    # 4. 반전 전 / 후 차이 계산
    original_distance = (
        mean_joint_distance(
            left_normalized,
            right_normalized,
        )
    )

    flipped_distance = (
        mean_joint_distance(
            flipped_left,
            right_normalized,
        )
    )

    output_path = (
        ROOT
        / "reports"
        / "day03"
        / "tables"
        / "left_right_flip_comparison.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for index, name in enumerate(
        COCO_KEYPOINT_NAMES
    ):
        left_x, left_y = (
            left_normalized[index]
        )

        flipped_x, flipped_y = (
            flipped_left[index]
        )

        right_x, right_y = (
            right_normalized[index]
        )

        original_joint_distance = (
            np.linalg.norm(
                left_normalized[index]
                - right_normalized[index]
            )
        )

        flipped_joint_distance = (
            np.linalg.norm(
                flipped_left[index]
                - right_normalized[index]
            )
        )

        rows.append(
            {
                "keypoint_id": index,
                "keypoint_name": name,

                "left_x": round(
                    float(left_x),
                    6,
                ),
                "left_y": round(
                    float(left_y),
                    6,
                ),

                "flipped_left_x": round(
                    float(flipped_x),
                    6,
                ),
                "flipped_left_y": round(
                    float(flipped_y),
                    6,
                ),

                "actual_right_x": round(
                    float(right_x),
                    6,
                ),
                "actual_right_y": round(
                    float(right_y),
                    6,
                ),

                "original_distance": round(
                    float(
                        original_joint_distance
                    ),
                    6,
                ),

                "flipped_distance": round(
                    float(
                        flipped_joint_distance
                    ),
                    6,
                ),
            }
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
        f"Left image : "
        f"{left_source.name}"
    )

    print(
        f"Right image: "
        f"{right_source.name}"
    )

    print()

    print(
        "Mean Joint Distance"
    )

    print(
        f"Original Left vs Right : "
        f"{original_distance:.4f}"
    )

    print(
        f"Flipped Left vs Right  : "
        f"{flipped_distance:.4f}"
    )

    print()

    print(
        "Selected Keypoints"
    )

    for index in [
        5, 6,
        7, 8,
        9, 10,
    ]:
        row = rows[index]

        print(
            f"{index:02d} "
            f"{row['keypoint_name']:16s} "
            f"left_x="
            f"{row['left_x']:7.3f} "
            f"flip_x="
            f"{row['flipped_left_x']:7.3f} "
            f"right_x="
            f"{row['actual_right_x']:7.3f}"
        )

    print()

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()