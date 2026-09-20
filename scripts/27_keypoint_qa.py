from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config


DEFAULT_MANIFEST_PATH = (
    ROOT
    / "data"
    / "day05"
    / "manifests"
    / "keypoint_dataset_manifest.csv"
)

DEFAULT_QA_PATH = (
    ROOT
    / "data"
    / "day05"
    / "qa"
    / "keypoint_clip_qa.csv"
)

DEFAULT_REVIEW_PATH = (
    ROOT
    / "data"
    / "day05"
    / "qa"
    / "keypoint_clip_review.csv"
)

ACTION_CRITICAL_KEYPOINTS = {
    "bend_return": [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
    ],
    "leg_raise_lower": [
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    "walk_turn_walk": [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    "drink_return": [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ],
    "sit_stand": [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    "wave": [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest-path",
        default=str(
            DEFAULT_MANIFEST_PATH
        ),
    )

    parser.add_argument(
        "--qa-path",
        default=str(
            DEFAULT_QA_PATH
        ),
    )

    parser.add_argument(
        "--review-path",
        default=str(
            DEFAULT_REVIEW_PATH
        ),
    )

    parser.add_argument(
        "--expected-clips",
        type=int,
        default=18,
    )

    return parser.parse_args()


def inspect_keypoint_csv(
    csv_path: Path,
    action_label: str,
    threshold: float,
) -> dict:
    df = pd.read_csv(csv_path)

    if df.empty:
        return {
            "qa_status": "CHECK",
            "frame_count": 0,
            "pose_ratio": 0.0,
            "single_person_ratio": 0.0,
            "mean_keypoint_conf": 0.0,
            "low_critical_ratio": 1.0,
            "missing_pose_frames": 0,
            "critical_keypoints": "",
        }

    critical_names = (
        ACTION_CRITICAL_KEYPOINTS.get(
            action_label
        )
    )

    if critical_names is None:
        raise RuntimeError(
            "QA 중요 Keypoint 정의가 "
            f"없는 Action: {action_label}"
        )

    frame_count = len(df)

    pose_series = (
        pd.to_numeric(
            df["pose_detected"],
            errors="coerce",
        )
        .fillna(0)
    )

    person_series = (
        pd.to_numeric(
            df["person_count"],
            errors="coerce",
        )
        .fillna(0)
    )

    pose_ratio = float(
        pose_series.mean()
    )

    single_person_ratio = float(
        (person_series == 1).mean()
    )

    confidence_columns = [
        column
        for column in df.columns
        if column.endswith("_conf")
        and column
        != "selected_box_conf"
    ]

    confidence_values = (
        df[confidence_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    mean_keypoint_conf = (
        float(
            np.nanmean(
                confidence_values
            )
        )
        if np.isfinite(
            confidence_values
        ).any()
        else 0.0
    )

    critical_columns = [
        f"{name}_conf"
        for name in critical_names
    ]

    missing_columns = [
        column
        for column in critical_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "중요 Keypoint Confidence 열이 "
            "없습니다: "
            + ", ".join(
                missing_columns
            )
        )

    critical_values = (
        df[critical_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    valid_mask = np.isfinite(
        critical_values
    )

    valid_count = int(
        valid_mask.sum()
    )

    if valid_count > 0:
        low_count = int(
            (
                (
                    critical_values
                    < threshold
                )
                & valid_mask
            ).sum()
        )

        low_critical_ratio = (
            low_count / valid_count
        )
    else:
        low_critical_ratio = 1.0

    missing_pose_frames = int(
        (pose_series == 0).sum()
    )

    qa_status = "AUTO_PASS"

    if pose_ratio < 0.90:
        qa_status = "CHECK"

    if single_person_ratio < 0.90:
        qa_status = "CHECK"

    if mean_keypoint_conf < 0.50:
        qa_status = "CHECK"

    if low_critical_ratio > 0.20:
        qa_status = "CHECK"

    return {
        "qa_status": qa_status,
        "frame_count": frame_count,
        "pose_ratio": round(
            pose_ratio,
            4,
        ),
        "single_person_ratio": round(
            single_person_ratio,
            4,
        ),
        "mean_keypoint_conf": round(
            mean_keypoint_conf,
            4,
        ),
        "low_critical_ratio": round(
            low_critical_ratio,
            4,
        ),
        "missing_pose_frames": (
            missing_pose_frames
        ),
        "critical_keypoints": "|".join(
            critical_names
        ),
    }


def main() -> None:
    args = parse_args()

    manifest_path = Path(
        args.manifest_path
    )
    qa_path = Path(
        args.qa_path
    )
    review_path = Path(
        args.review_path
    )

    if args.expected_clips < 1:
        raise ValueError(
            "--expected-clips는 "
            "1 이상이어야 합니다."
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            manifest_path
        )

    config = load_config()

    threshold = float(
        config.get(
            "keypoint_confidence",
            0.50,
        )
    )

    manifest = pd.read_csv(
        manifest_path
    )

    if len(manifest) != args.expected_clips:
        raise RuntimeError(
            "Manifest Clip 수가 "
            "예상과 다릅니다. "
            f"expected={args.expected_clips}, "
            f"actual={len(manifest)}"
        )

    rows = []

    for _, item in (
        manifest.iterrows()
    ):
        csv_path = (
            ROOT
            / str(
                item["keypoint_csv"]
            )
        )

        if not csv_path.exists():
            raise FileNotFoundError(
                csv_path
            )

        qa = inspect_keypoint_csv(
            csv_path=csv_path,
            action_label=str(
                item["action_label"]
            ),
            threshold=threshold,
        )

        row = {
            "clip_id": item["clip_id"],
            "subject": item["subject"],
            "action_id": item["action_id"],
            "action_label": (
                item["action_label"]
            ),
            "take": item["take"],
            "video_path": (
                item["video_path"]
            ),
            "keypoint_csv": (
                item["keypoint_csv"]
            ),
            **qa,
            "human_decision": (
                "UNREVIEWED"
            ),
            "human_note": "",
        }

        rows.append(row)

        print(
            f"{item['clip_id']:16s} "
            f"{qa['qa_status']:9s} "
            f"pose={qa['pose_ratio']:.3f} "
            f"conf="
            f"{qa['mean_keypoint_conf']:.3f} "
            f"low="
            f"{qa['low_critical_ratio']:.3f}"
        )

    qa_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

    for output_path in [
        qa_path,
        review_path,
    ]:
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

    print()
    print(f"QA     : {qa_path}")
    print(f"Review : {review_path}")


if __name__ == "__main__":
    main()