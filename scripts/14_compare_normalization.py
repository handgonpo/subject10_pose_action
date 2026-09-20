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
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--a",
        default=str(
            ROOT
            / "data"
            / "day03"
            / "images"
            / "arms_up_near.jpg"
        ),
    )

    parser.add_argument(
        "--b",
        default=str(
            ROOT
            / "data"
            / "day03"
            / "images"
            / "arms_up_far.jpg"
        ),
    )

    parser.add_argument(
        "--output",
        default="normalization_comparison.csv",
        help=(
            "reports/day03/tables 아래에 "
            "저장할 CSV 파일명"
        ),
    )

    return parser.parse_args()


def extract_first_person(
    model,
    source,
    config,
    device,
):
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
            f"Pose 검출 실패: "
            f"{source}"
        )

    keypoints = (
        result.keypoints.xy[0]
        .cpu()
        .numpy()
    )

    return keypoints


def mean_joint_distance(
    pose_a,
    pose_b,
):
    distances = np.linalg.norm(
        pose_a - pose_b,
        axis=1,
    )

    return float(
        np.mean(distances)
    )


def main() -> None:
    args = parse_args()

    source_a = Path(args.a)
    source_b = Path(args.b)

    if not source_a.exists():
        raise FileNotFoundError(
            source_a
        )

    if not source_b.exists():
        raise FileNotFoundError(
            source_b
        )

    config = load_config()

    device = resolve_device(
        config["device"]
    )

    model = YOLO(
        config["pose_model"]
    )

    keypoints_a = (
        extract_first_person(
            model,
            source_a,
            config,
            device,
        )
    )

    keypoints_b = (
        extract_first_person(
            model,
            source_b,
            config,
            device,
        )
    )

    normalized_a, (
        center_a
    ), scale_a = (
        normalize_by_shoulder_width(
            keypoints_a
        )
    )

    normalized_b, (
        center_b
    ), scale_b = (
        normalize_by_shoulder_width(
            keypoints_b
        )
    )

    pixel_difference = (
        mean_joint_distance(
            keypoints_a,
            keypoints_b,
        )
    )

    normalized_difference = (
        mean_joint_distance(
            normalized_a,
            normalized_b,
        )
    )

    print(
        f"A: {source_a.name}"
    )

    print(
        f"B: {source_b.name}"
    )

    print()

    print(
        "Mean joint distance"
    )

    print(
        f"Pixel space      : "
        f"{pixel_difference:.4f}"
    )

    print(
        f"Normalized space : "
        f"{normalized_difference:.4f}"
    )

    print()

    print(
        f"A hip center     : "
        f"{center_a}"
    )

    print(
        f"B hip center     : "
        f"{center_b}"
    )

    print(
        f"A shoulder width : "
        f"{scale_a:.2f}"
    )

    print(
        f"B shoulder width : "
        f"{scale_b:.2f}"
    )

    output_path = (
        ROOT
        / "reports"
        / "day03"
        / "tables"
        / args.output
    )

    rows = [
        {
            "pair": (
                f"{source_a.name}"
                f" vs "
                f"{source_b.name}"
            ),
            "pixel_mean_distance": (
                round(
                    pixel_difference,
                    6,
                )
            ),
            "normalized_mean_distance": (
                round(
                    normalized_difference,
                    6,
                )
            ),
            "a_shoulder_width": (
                round(
                    scale_a,
                    4,
                )
            ),
            "b_shoulder_width": (
                round(
                    scale_b,
                    4,
                )
            ),
        }
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(rows)

    print()
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()