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
    body_center,
    normalize_by_shoulder_width,
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
            "Keypoint를 찾지 "
            "못했습니다."
        )
        return

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

    center = body_center(
        keypoints
    )

    normalized, (
        used_center
    ), scale = (
        normalize_by_shoulder_width(
            keypoints
        )
    )

    centered = (
        keypoints
        - used_center
    )

    output_path = (
        ROOT
        / "reports"
        / "day03"
        / "tables"
        / (
            f"{source.stem}"
            "_normalization.csv"
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "keypoint_id",
        "keypoint_name",
        "confidence",
        "pixel_x",
        "pixel_y",
        "centered_x",
        "centered_y",
        "normalized_x",
        "normalized_y",
    ]

    rows = []

    for keypoint_id, name in (
        enumerate(
            COCO_KEYPOINT_NAMES
        )
    ):
        x, y = keypoints[
            keypoint_id
        ]

        cx, cy = centered[
            keypoint_id
        ]

        nx, ny = normalized[
            keypoint_id
        ]

        conf = confidence[
            keypoint_id
        ]

        rows.append(
            {
                "keypoint_id": (
                    keypoint_id
                ),
                "keypoint_name": name,
                "confidence": round(
                    float(conf),
                    4,
                ),
                "pixel_x": round(
                    float(x),
                    2,
                ),
                "pixel_y": round(
                    float(y),
                    2,
                ),
                "centered_x": round(
                    float(cx),
                    4,
                ),
                "centered_y": round(
                    float(cy),
                    4,
                ),
                "normalized_x": round(
                    float(nx),
                    6,
                ),
                "normalized_y": round(
                    float(ny),
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
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)

    print(
        f"Source: {source.name}"
    )

    print(
        "Hip center:",
        f"({center[0]:.2f}, "
        f"{center[1]:.2f})"
    )

    print(
        f"Shoulder width: "
        f"{scale:.2f}"
    )

    print()
    print(
        "Selected Keypoints"
    )

    for index in [
        5, 6,
        9, 10,
        11, 12,
        15, 16,
    ]:
        name = (
            COCO_KEYPOINT_NAMES[
                index
            ]
        )

        x, y = keypoints[index]

        nx, ny = normalized[
            index
        ]

        print(
            f"{index:02d} "
            f"{name:16s} "
            f"pixel=({x:7.1f}, "
            f"{y:7.1f}) "
            f"normalized="
            f"({nx:7.3f}, "
            f"{ny:7.3f})"
        )

    print()
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()