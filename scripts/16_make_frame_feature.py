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
    normalize_by_shoulder_width,
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


def make_feature_names():
    names = []

    for keypoint_name in (
        COCO_KEYPOINT_NAMES
    ):
        names.append(
            f"{keypoint_name}_x"
        )

        names.append(
            f"{keypoint_name}_y"
        )

    names.extend(
        [
            "left_elbow_angle",
            "right_elbow_angle",
            "left_knee_angle",
            "right_knee_angle",
        ]
    )

    return names


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

    normalized, (
        center
    ), scale = (
        normalize_by_shoulder_width(
            keypoints
        )
    )

    angles = selected_joint_angles(
        keypoints
    )

    feature_values = []

    for point in normalized:
        feature_values.extend(
            [
                float(point[0]),
                float(point[1]),
            ]
        )

    feature_values.extend(
        [
            angles[
                "left_elbow_angle"
            ],
            angles[
                "right_elbow_angle"
            ],
            angles[
                "left_knee_angle"
            ],
            angles[
                "right_knee_angle"
            ],
        ]
    )

    feature_names = (
        make_feature_names()
    )

    mean_confidence = float(
        np.mean(
            confidence
        )
    )

    low_keypoint_count = int(
        np.sum(
            confidence
            < args.kp_conf
        )
    )

    row = {
        "source": source.name,
        "mean_keypoint_conf": (
            round(
                mean_confidence,
                4,
            )
        ),
        "low_keypoint_count": (
            low_keypoint_count
        ),
        "hip_center_x": round(
            float(center[0]),
            4,
        ),
        "hip_center_y": round(
            float(center[1]),
            4,
        ),
        "shoulder_width": round(
            float(scale),
            4,
        ),
    }

    for name, value in zip(
        feature_names,
        feature_values,
    ):
        row[name] = round(
            float(value),
            6,
        )

    output_path = (
        ROOT
        / "reports"
        / "day03"
        / "tables"
        / (
            f"{source.stem}"
            "_frame_feature.csv"
        )
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                row.keys()
            ),
        )

        writer.writeheader()

        writer.writerow(row)

    print(
        f"Source: {source.name}"
    )

    print(
        f"Mean keypoint confidence: "
        f"{mean_confidence:.3f}"
    )

    print(
        f"Low keypoint count: "
        f"{low_keypoint_count}"
    )

    print(
        f"Feature count: "
        f"{len(feature_values)}"
    )

    print()

    print(
        "Joint angles"
    )

    for name, value in (
        angles.items()
    ):
        print(
            f"{name:20s}: "
            f"{value:7.2f}°"
        )

    print()

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()