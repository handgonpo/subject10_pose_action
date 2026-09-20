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
from src.pose_utils import (
    COCO_SKELETON,
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
    )

    parser.add_argument(
        "--kp-conf",
        type=float,
        default=0.50,
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
        result.keypoints is None
        or result.keypoints.xy is None
        or len(result.keypoints.xy) == 0
    ):
        print("Keypoint를 찾지 못했습니다.")
        return

    xy = result.keypoints.xy.cpu().numpy()

    if result.keypoints.conf is not None:
        confidence = (
            result.keypoints.conf
            .cpu()
            .numpy()
        )
    else:
        confidence = np.ones(
            xy.shape[:2],
            dtype=float,
        )

    output = image.copy()

    if result.boxes is not None:
        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

    for person_index in range(len(xy)):
        points = xy[person_index]
        confs = confidence[person_index]

        for start_id, end_id in COCO_SKELETON:
            if (
                confs[start_id] >= args.kp_conf
                and confs[end_id] >= args.kp_conf
            ):
                start = tuple(
                    points[start_id].astype(int)
                )
                end = tuple(
                    points[end_id].astype(int)
                )

                cv2.line(
                    output,
                    start,
                    end,
                    (0, 255, 0),
                    2,
                )

        for keypoint_id, (point, conf) in enumerate(
            zip(points, confs)
        ):
            if conf < args.kp_conf:
                continue

            x, y = point.astype(int)

            cv2.circle(
                output,
                (x, y),
                5,
                (0, 255, 255),
                -1,
            )

            cv2.putText(
                output,
                str(keypoint_id),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

    threshold_name = int(
        args.kp_conf * 100
    )

    output_path = (
        ROOT
        / "reports"
        / "day02"
        / "images"
        / (
            f"{source.stem}_"
            f"skeleton_conf{threshold_name:02d}.jpg"
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = cv2.imwrite(
        str(output_path),
        output,
    )

    if not saved:
        raise RuntimeError(
            f"결과를 저장하지 못했습니다: {output_path}"
        )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()