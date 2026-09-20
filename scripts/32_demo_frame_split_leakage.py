from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import (
    train_test_split,
)


ROOT = Path(__file__).resolve().parents[1]

KEYPOINT_ROOT = (
    ROOT
    / "data"
    / "day05"
    / "keypoints"
)


def main() -> None:
    csv_paths = sorted(
        KEYPOINT_ROOT.rglob("*.csv")
    )

    if len(csv_paths) != 18:
        raise RuntimeError(
            "기본 Keypoint CSV는 "
            f"18개여야 합니다. "
            f"현재 {len(csv_paths)}개입니다."
        )

    frames = []

    for path in csv_paths:
        df = pd.read_csv(
            path,
            usecols=[
                "clip_id",
                "frame_id",
                "subject",
                "action_label",
            ],
        )
        frames.append(df)

    all_frames = pd.concat(
        frames,
        ignore_index=True,
    )

    train_df, test_df = (
        train_test_split(
            all_frames,
            test_size=0.20,
            random_state=42,
            shuffle=True,
        )
    )

    train_subjects = set(
        train_df["subject"]
        .astype(str)
    )
    test_subjects = set(
        test_df["subject"]
        .astype(str)
    )

    subject_overlap = (
        train_subjects
        & test_subjects
    )

    train_clips = set(
        train_df["clip_id"]
        .astype(str)
    )
    test_clips = set(
        test_df["clip_id"]
        .astype(str)
    )

    clip_overlap = (
        train_clips
        & test_clips
    )

    print("Random Frame Split Demo")
    print()
    print(
        f"Train frames: "
        f"{len(train_df)}"
    )
    print(
        f"Test frames : "
        f"{len(test_df)}"
    )
    print()
    print(
        "Subject overlap:",
        sorted(subject_overlap),
    )
    print(
        "Clip overlap count:",
        len(clip_overlap),
    )
    print()

    if subject_overlap:
        print(
            "[LEAKAGE] 같은 Subject가 "
            "Train과 Test에 존재"
        )

    if clip_overlap:
        print(
            "[LEAKAGE] 같은 Clip의 "
            "Frame이 Train과 Test에 존재"
        )

    print()
    print(
        "이 Random Frame Split은 "
        "최종 평가용으로 "
        "사용하지 않습니다."
    )


if __name__ == "__main__":
    main()