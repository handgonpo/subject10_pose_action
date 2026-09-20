from __future__ import annotations

import argparse
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


COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help="Pose를 적용할 이미지 파일",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = Path(args.source)

    if not source.exists():
        raise FileNotFoundError(
            f"파일이 없습니다: {source}"
        )

    config = load_config()
    device = resolve_device(config["device"])

    model = YOLO(config["pose_model"])

    results = model.predict(
        source=str(source),
        imgsz=config["image_size"],
        conf=config["person_confidence"],
        device=device,
        verbose=False,
    )

    result = results[0]

    person_count = (
        len(result.boxes)
        if result.boxes is not None
        else 0
    )

    print(f"Detected people: {person_count}")

    output_image = result.plot()

    output_path = (
        ROOT
        / "reports"
        / "day01"
        / f"pose_{source.stem}.jpg"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = cv2.imwrite(
        str(output_path),
        output_image,
    )

    if not saved:
        raise RuntimeError(
            f"결과 이미지를 저장하지 못했습니다: {output_path}"
        )

    print(f"Saved image: {output_path}")

    if (
        result.keypoints is None
        or result.keypoints.xy is None
        or person_count == 0
    ):
        print("Keypoint가 검출되지 않았습니다.")
        return

    xy = result.keypoints.xy.cpu().numpy()

    confidence = result.keypoints.conf

    if confidence is not None:
        confidence = confidence.cpu().numpy()
    else:
        confidence = np.ones(
            xy.shape[:2],
            dtype=float,
        )

    report_lines = []

    for person_index in range(len(xy)):
        title = f"Person {person_index + 1}"

        print(f"\n{title}")
        report_lines.append(title)

        for keypoint_index, name in enumerate(
            COCO_KEYPOINT_NAMES
        ):
            x, y = xy[
                person_index,
                keypoint_index,
            ]

            conf = confidence[
                person_index,
                keypoint_index,
            ]

            line = (
                f"{keypoint_index:02d} "
                f"{name:16s} "
                f"x={x:8.2f} "
                f"y={y:8.2f} "
                f"conf={conf:.3f}"
            )

            print(line)
            report_lines.append(line)

    report_path = (
        ROOT
        / "reports"
        / "day01"
        / f"pose_{source.stem}.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(f"\nKeypoint report: {report_path}")


if __name__ == "__main__":
    main()