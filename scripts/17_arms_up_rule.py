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
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_WRIST,
    RIGHT_WRIST,
    normalize_by_shoulder_width,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
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

    result = model.predict(
        source=str(source),
        imgsz=config["image_size"],
        conf=config[
            "person_confidence"
        ],
        device=device,
        verbose=False,
    )[0]

    if (
        result.keypoints is None
        or result.keypoints.xy is None
        or len(
            result.keypoints.xy
        ) == 0
    ):
        raise RuntimeError(
            "Pose를 찾지 못했습니다."
        )

    keypoints = (
        result.keypoints.xy[0]
        .cpu()
        .numpy()
    )

    if (
        result.keypoints.conf
        is not None
    ):
        confidence = (
            result.keypoints.conf[0]
            .cpu()
            .numpy()
        )
    else:
        confidence = np.ones(
            17,
            dtype=np.float32,
        )

    required_ids = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_WRIST,
        RIGHT_WRIST,
    ]

    min_conf = min(
        float(
            confidence[index]
        )
        for index in required_ids
    )

    if min_conf < args.kp_conf:
        print(
            f"SKIP: 필요한 관절의 "
            f"최소 Confidence가 "
            f"{min_conf:.3f}입니다."
        )
        return

    normalized, _, _ = (
        normalize_by_shoulder_width(
            keypoints
        )
    )

    left_wrist_y = (
        normalized[
            LEFT_WRIST,
            1,
        ]
    )

    right_wrist_y = (
        normalized[
            RIGHT_WRIST,
            1,
        ]
    )

    left_shoulder_y = (
        normalized[
            LEFT_SHOULDER,
            1,
        ]
    )

    right_shoulder_y = (
        normalized[
            RIGHT_SHOULDER,
            1,
        ]
    )

    left_above = (
        left_wrist_y
        < left_shoulder_y
    )

    right_above = (
        right_wrist_y
        < right_shoulder_y
    )

    arms_up = (
        left_above
        and right_above
    )

    print(
        f"Source: {source.name}"
    )

    print(
        f"Min required confidence: "
        f"{min_conf:.3f}"
    )

    print(
        f"Left wrist - shoulder: "
        f"{left_wrist_y - left_shoulder_y:.3f}"
    )

    print(
        f"Right wrist - shoulder: "
        f"{right_wrist_y - right_shoulder_y:.3f}"
    )

    print(
        f"Arms Up: {arms_up}"
    )


if __name__ == "__main__":
    main()