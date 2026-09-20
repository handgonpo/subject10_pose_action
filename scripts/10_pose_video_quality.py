from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (
    load_config,
    resolve_device,
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
            / "day02"
            / "videos"
            / "pose_quality_demo.mp4"
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
        help="처리할 최대 Frame 수. 0이면 영상 끝까지 처리",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = Path(args.source)

    if not source.exists():
        raise FileNotFoundError(
            f"영상이 없습니다: {source}"
        )

    config = load_config()
    device = resolve_device(config["device"])

    model = YOLO(config["pose_model"])

    capture = cv2.VideoCapture(str(source))

    if not capture.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: {source}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    output_fps = fps if fps > 0 else 30.0

    success, frame = capture.read()

    if not success:
        capture.release()
        raise RuntimeError(
            "영상의 첫 Frame을 읽지 못했습니다."
        )

    height, width = frame.shape[:2]

    output_video = (
        ROOT
        / "reports"
        / "day02"
        / "videos"
        / f"{source.stem}_pose.mp4"
    )

    output_video.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            "결과 MP4를 생성할 수 없습니다."
        )

    frame_table = (
        ROOT
        / "reports"
        / "day02"
        / "tables"
        / f"{source.stem}_frame_quality.csv"
    )

    keypoint_table = (
        ROOT
        / "reports"
        / "day02"
        / "tables"
        / f"{source.stem}_keypoints.csv"
    )

    frame_table.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_rows = []
    keypoint_rows = []

    frame_id = 0

    while success:
        frame_id += 1

        results = model.predict(
            source=frame,
            imgsz=config["image_size"],
            conf=config["person_confidence"],
            device=device,
            verbose=False,
        )

        result = results[0]

        writer.write(result.plot())

        people = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        mean_kp_conf = 0.0
        low_keypoints = 17
        lowest_name = "NO_PERSON"
        lowest_conf = 0.0

        if (
            result.boxes is not None
            and result.keypoints is not None
            and result.keypoints.xy is not None
            and result.keypoints.conf is not None
            and len(result.boxes) > 0
        ):
            box_conf = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            xy = (
                result.keypoints.xy
                .cpu()
                .numpy()
            )

            conf = (
                result.keypoints.conf
                .cpu()
                .numpy()
            )

            main_index = int(
                np.argmax(box_conf)
            )

            main_xy = xy[main_index]
            main_conf = conf[main_index]

            mean_kp_conf = float(
                np.mean(main_conf)
            )

            low_keypoints = int(
                np.sum(
                    main_conf < args.kp_conf
                )
            )

            lowest_id = int(
                np.argmin(main_conf)
            )

            lowest_name = (
                COCO_KEYPOINT_NAMES[
                    lowest_id
                ]
            )

            lowest_conf = float(
                main_conf[lowest_id]
            )

            for keypoint_id, name in enumerate(
                COCO_KEYPOINT_NAMES
            ):
                x, y = main_xy[keypoint_id]
                confidence = main_conf[keypoint_id]

                keypoint_rows.append(
                    {
                        "frame_id": frame_id,
                        "time_sec": round(
                            (frame_id - 1)
                            / output_fps,
                            3,
                        ),
                        "keypoint_id": keypoint_id,
                        "keypoint_name": name,
                        "pixel_x": round(
                            float(x),
                            2,
                        ),
                        "pixel_y": round(
                            float(y),
                            2,
                        ),
                        "confidence": round(
                            float(confidence),
                            4,
                        ),
                    }
                )

        frame_rows.append(
            {
                "frame_id": frame_id,
                "time_sec": round(
                    (frame_id - 1)
                    / output_fps,
                    3,
                ),
                "people": people,
                "mean_keypoint_conf": round(
                    mean_kp_conf,
                    4,
                ),
                "low_keypoint_count": (
                    low_keypoints
                ),
                "lowest_keypoint": (
                    lowest_name
                ),
                "lowest_keypoint_conf": round(
                    lowest_conf,
                    4,
                ),
                "keypoint_threshold": (
                    args.kp_conf
                ),
            }
        )

        if frame_id % 30 == 0:
            print(
                f"Processed frame: {frame_id}"
            )

        if (
            args.max_frames > 0
            and frame_id >= args.max_frames
        ):
            break

        success, frame = capture.read()

    capture.release()
    writer.release()

    with frame_table.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer_csv = csv.DictWriter(
            file,
            fieldnames=[
                "frame_id",
                "time_sec",
                "people",
                "mean_keypoint_conf",
                "low_keypoint_count",
                "lowest_keypoint",
                "lowest_keypoint_conf",
                "keypoint_threshold",
            ],
        )

        writer_csv.writeheader()
        writer_csv.writerows(frame_rows)

    with keypoint_table.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer_csv = csv.DictWriter(
            file,
            fieldnames=[
                "frame_id",
                "time_sec",
                "keypoint_id",
                "keypoint_name",
                "pixel_x",
                "pixel_y",
                "confidence",
            ],
        )

        writer_csv.writeheader()
        writer_csv.writerows(keypoint_rows)

    print()
    print(f"Frames processed: {frame_id}")
    print(f"Pose video: {output_video}")
    print(f"Frame table: {frame_table}")
    print(f"Keypoint table: {keypoint_table}")


if __name__ == "__main__":
    main()