from __future__ import annotations

import argparse
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
    selected_joint_angles,
)

from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default=str(
            ROOT
            / "data"
            / "day03"
            / "images"
            / "arms_up_person_a.jpg"
        ),
    )

    parser.add_argument(
        "--kp-conf",
        type=float,
        default=0.50,
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

    config = load_config()

    device = resolve_device(
        config["device"]
    )

    model = YOLO(
        config["pose_model"]
    )

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
        or result.keypoints.xy
        is None
        or len(
            result.keypoints.xy
        ) == 0
    ):
        print(
            "사람의 Keypoint를 "
            "찾지 못했습니다."
        )
        return

    xy = (
        result.keypoints.xy
        .cpu()
        .numpy()
    )

    if (
        result.keypoints.conf
        is not None
    ):
        confidence = (
            result.keypoints.conf
            .cpu()
            .numpy()
        )
    else:
        confidence = np.ones(
            xy.shape[:2],
            dtype=np.float32,
        )

    # 오늘의 기본 실습에서는
    # 첫 번째 사람을 사용합니다.
    person_xy = xy[0]

    person_conf = confidence[0]

    print(
        f"Source: {source.name}"
    )

    print()

    print(
        "관절 Confidence 확인"
    )

    for keypoint_id in [
        5, 7, 9,
        6, 8, 10,
        11, 13, 15,
        12, 14, 16,
    ]:
        print(
            f"{keypoint_id:02d} "
            f"{COCO_KEYPOINT_NAMES[keypoint_id]:16s} "
            f"{person_conf[keypoint_id]:.3f}"
        )

    angle_specs = {
        "left_elbow_angle": (
            5, 7, 9
        ),

        "right_elbow_angle": (
            6, 8, 10
        ),

        "left_knee_angle": (
            11, 13, 15
        ),

        "right_knee_angle": (
            12, 14, 16
        ),
    }

    angles = selected_joint_angles(
        person_xy
    )

    print()
    print(
        "Joint Angles"
    )

    for name, angle in (
        angles.items()
    ):
        required_ids = (
            angle_specs[name]
        )

        required_conf = [
            person_conf[index]
            for index
            in required_ids
        ]

        min_conf = min(
            required_conf
        )

        if min_conf < args.kp_conf:
            print(
                f"{name:20s}: "
                f"SKIP "
                f"(min conf={min_conf:.3f})"
            )

            continue

        print(
            f"{name:20s}: "
            f"{angle:7.2f}°"
        )


if __name__ == "__main__":
    main()