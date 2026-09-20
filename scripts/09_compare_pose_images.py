from __future__ import annotations

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


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def main() -> None:
    config = load_config()
    device = resolve_device(config["device"])

    kp_threshold = config.get(
        "keypoint_confidence",
        0.50,
    )

    image_dir = (
        ROOT
        / "data"
        / "day02"
        / "images"
    )

    output_dir = (
        ROOT
        / "reports"
        / "day02"
        / "images"
        / "comparison"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_path = (
        ROOT
        / "reports"
        / "day02"
        / "tables"
        / "image_quality_summary.csv"
    )

    table_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower()
        in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            f"이미지가 없습니다: {image_dir}"
        )

    model = YOLO(config["pose_model"])

    rows = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skip: {image_path}")
            continue

        height, width = image.shape[:2]
        image_area = width * height

        results = model.predict(
            source=image,
            imgsz=config["image_size"],
            conf=config["person_confidence"],
            device=device,
            verbose=False,
        )

        result = results[0]

        people = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        main_box_conf = 0.0
        bbox_area_ratio = 0.0
        mean_kp_conf = 0.0
        low_keypoints = 17
        lowest_name = "NO_PERSON"
        lowest_conf = 0.0

        if (
            result.boxes is not None
            and result.keypoints is not None
            and result.keypoints.conf is not None
            and len(result.boxes) > 0
        ):
            box_conf = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            kp_conf = (
                result.keypoints.conf
                .cpu()
                .numpy()
            )

            main_index = int(
                np.argmax(box_conf)
            )

            main_box_conf = float(
                box_conf[main_index]
            )

            x1, y1, x2, y2 = boxes[main_index]

            bbox_area = max(
                0.0,
                float(x2 - x1),
            ) * max(
                0.0,
                float(y2 - y1),
            )

            if image_area > 0:
                bbox_area_ratio = (
                    bbox_area / image_area
                )

            conf = kp_conf[main_index]

            mean_kp_conf = float(
                np.mean(conf)
            )

            low_keypoints = int(
                np.sum(
                    conf < kp_threshold
                )
            )

            lowest_id = int(
                np.argmin(conf)
            )

            lowest_name = (
                COCO_KEYPOINT_NAMES[
                    lowest_id
                ]
            )

            lowest_conf = float(
                conf[lowest_id]
            )

        output_path = (
            output_dir
            / f"{image_path.stem}_pose.jpg"
        )

        saved = cv2.imwrite(
            str(output_path),
            result.plot(),
        )

        if not saved:
            raise RuntimeError(
                f"결과 이미지를 저장하지 못했습니다: {output_path}"
            )

        rows.append(
            {
                "image": image_path.name,
                "people": people,
                "main_box_conf": round(
                    main_box_conf,
                    4,
                ),
                "bbox_area_ratio": round(
                    bbox_area_ratio,
                    4,
                ),
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
                    kp_threshold
                ),
            }
        )

        print(
            image_path.name,
            "people=",
            people,
            "bbox_ratio=",
            f"{bbox_area_ratio:.3f}",
            "mean_kp=",
            f"{mean_kp_conf:.3f}",
            "low_kp=",
            low_keypoints,
        )

    with table_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "people",
                "main_box_conf",
                "bbox_area_ratio",
                "mean_keypoint_conf",
                "low_keypoint_count",
                "lowest_keypoint",
                "lowest_keypoint_conf",
                "keypoint_threshold",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Summary saved: {table_path}")


if __name__ == "__main__":
    main()