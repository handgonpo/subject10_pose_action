from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
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
            / "videos"
            / "feature_demo.mp4"
        ),
    )

    parser.add_argument(
        "--kp-conf",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="처리할 최대 Frame 수, 0이면 전체 처리",
    )

    return parser.parse_args()


def feature_names():
    names = []

    for name in (
        COCO_KEYPOINT_NAMES
    ):
        names.extend(
            [
                f"{name}_x",
                f"{name}_y",
            ]
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

    capture = cv2.VideoCapture(
        str(source)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: "
            f"{source}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    names = feature_names()

    output_path = (
        ROOT
        / "reports"
        / "day03"
        / "tables"
        / (
            f"{source.stem}"
            "_frame_features.csv"
        )
    )

    fieldnames = [
        "frame_id",
        "time_sec",
        "valid_pose",
        "mean_keypoint_conf",
        "low_keypoint_count",
    ] + names

    rows = []

    frame_id = 0

    while True:
        if (
            args.max_frames > 0
            and frame_id >= args.max_frames
        ):
            break

        success, frame = (
            capture.read()
        )

        if not success:
            break

        frame_id += 1

        result = model.predict(
            source=frame,
            imgsz=config["image_size"],
            conf=config[
                "person_confidence"
            ],
            device=device,
            verbose=False,
        )[0]

        base_row = {
            "frame_id": frame_id,
            "time_sec": round(
                (frame_id - 1) / fps,
                4,
            ),
            "valid_pose": 0,
            "mean_keypoint_conf": (
                0.0
            ),
            "low_keypoint_count": 17,
        }

        for name in names:
            base_row[name] = ""

        if (
            result.keypoints
            is not None
            and result.keypoints.xy
            is not None
            and len(
                result.keypoints.xy
            ) > 0
        ):
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
                confidence = (
                    np.ones(
                        17,
                        dtype=np.float32,
                    )
                )

            mean_conf = float(
                np.mean(
                    confidence
                )
            )

            low_count = int(
                np.sum(
                    confidence
                    < args.kp_conf
                )
            )

            try:
                normalized, _, _ = (
                    normalize_by_shoulder_width(
                        keypoints
                    )
                )

                angles = (
                    selected_joint_angles(
                        keypoints
                    )
                )

                values = []

                for point in normalized:
                    values.extend(
                        [
                            float(
                                point[0]
                            ),
                            float(
                                point[1]
                            ),
                        ]
                    )

                values.extend(
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

                base_row[
                    "valid_pose"
                ] = 1

                base_row[
                    "mean_keypoint_conf"
                ] = round(
                    mean_conf,
                    4,
                )

                base_row[
                    "low_keypoint_count"
                ] = low_count

                for name, value in zip(
                    names,
                    values,
                ):
                    base_row[name] = (
                        round(
                            float(value),
                            6,
                        )
                    )

            except ValueError:
                pass

        rows.append(
            base_row
        )

        if frame_id % 30 == 0:
            print(
                f"Processed: "
                f"{frame_id}"
            )

    capture.release()

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)

    valid_count = sum(
        int(
            row["valid_pose"]
        )
        for row in rows
    )

    print()
    print(
        f"Total frames: "
        f"{frame_id}"
    )

    print(
        f"Valid pose frames: "
        f"{valid_count}"
    )

    print(
        f"Feature count per frame: "
        f"{len(names)}"
    )

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()