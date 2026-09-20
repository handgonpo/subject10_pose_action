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
    confidence_state,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default=str(
            ROOT
            / "data"
            / "day02"
            / "images"
            / "front.jpg"
        ),
        help="분석할 이미지 파일",
    )

    parser.add_argument(
        "--kp-conf",
        type=float,
        default=0.50,
        help="VALID로 판단할 Keypoint Confidence 기준",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = Path(args.source)

    if not source.exists():
        raise FileNotFoundError(
            f"이미지가 없습니다: {source}"
        )

    image = cv2.imread(str(source))

    if image is None:
        raise RuntimeError(
            f"이미지를 읽지 못했습니다: {source}"
        )

    height, width = image.shape[:2]

    config = load_config()
    device = resolve_device(config["device"])

    model = YOLO(config["pose_model"])

    results = model.predict(
        source=image,
        imgsz=config["image_size"],
        conf=config["person_confidence"],
        device=device,
        verbose=False,
    )

    result = results[0]

    if (
        result.boxes is None
        or result.keypoints is None
        or result.keypoints.xy is None
        or len(result.boxes) == 0
    ):
        print("사람 또는 Keypoint를 찾지 못했습니다.")
        return

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    boxes_conf = result.boxes.conf.cpu().numpy()
    xy = result.keypoints.xy.cpu().numpy()

    if result.keypoints.conf is not None:
        kp_conf = result.keypoints.conf.cpu().numpy()
    else:
        kp_conf = np.ones(
            xy.shape[:2],
            dtype=float,
        )

    output_image = (
        ROOT
        / "reports"
        / "day02"
        / "images"
        / f"{source.stem}_pose.jpg"
    )

    output_image.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = cv2.imwrite(
        str(output_image),
        result.plot(),
    )

    if not saved:
        raise RuntimeError(
            f"결과 이미지를 저장하지 못했습니다: {output_image}"
        )

    output_csv = (
        ROOT
        / "reports"
        / "day02"
        / "tables"
        / f"{source.stem}_keypoints.csv"
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "person_id",
        "box_conf",
        "keypoint_id",
        "keypoint_name",
        "pixel_x",
        "pixel_y",
        "normalized_x",
        "normalized_y",
        "keypoint_conf",
        "quality",
    ]

    rows = []

    person_count = min(
        len(boxes_xyxy),
        len(xy),
    )

    print(f"Detected people: {person_count}")
    print(f"Image size: {width} x {height}")

    for person_index in range(person_count):
        x1, y1, x2, y2 = boxes_xyxy[person_index]

        print()
        print(f"Person {person_index + 1}")
        print(
            "BBox:",
            f"({x1:.1f}, {y1:.1f})",
            f"({x2:.1f}, {y2:.1f})",
        )
        print(
            f"BBox Confidence: "
            f"{boxes_conf[person_index]:.3f}"
        )

        for keypoint_id, name in enumerate(
            COCO_KEYPOINT_NAMES
        ):
            x, y = xy[
                person_index,
                keypoint_id,
            ]

            confidence = kp_conf[
                person_index,
                keypoint_id,
            ]

            normalized_x = (
                float(x) / width
                if width > 0
                else 0.0
            )

            normalized_y = (
                float(y) / height
                if height > 0
                else 0.0
            )

            quality = confidence_state(
                float(confidence),
                valid_threshold=args.kp_conf,
                warning_threshold=0.30,
            )

            row = {
                "person_id": person_index + 1,
                "box_conf": round(
                    float(boxes_conf[person_index]),
                    4,
                ),
                "keypoint_id": keypoint_id,
                "keypoint_name": name,
                "pixel_x": round(float(x), 2),
                "pixel_y": round(float(y), 2),
                "normalized_x": round(
                    normalized_x,
                    5,
                ),
                "normalized_y": round(
                    normalized_y,
                    5,
                ),
                "keypoint_conf": round(
                    float(confidence),
                    4,
                ),
                "quality": quality,
            }

            rows.append(row)

            print(
                f"{keypoint_id:02d} "
                f"{name:16s} "
                f"pixel=({x:7.1f}, {y:7.1f}) "
                f"norm=({normalized_x:.4f}, {normalized_y:.4f}) "
                f"conf={confidence:.3f} "
                f"{quality}"
            )

    with output_csv.open(
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

    print()
    print(f"Pose image: {output_image}")
    print(f"Keypoint CSV: {output_csv}")


if __name__ == "__main__":
    main()