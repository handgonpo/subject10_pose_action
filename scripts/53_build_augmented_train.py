from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.augmentation_utils import (
    horizontal_flip_dataframe,
    jitter_dataframe,
)
from src.sequence_utils import (
    build_frame_feature_matrix,
    flatten_sequence,
    resample_sequence,
    sequence_feature_names,
)


TRAIN_SPLIT = (
    ROOT / "data" / "day05" / "splits" / "train.csv"
)
TRAIN_INFO_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_training_info.json"
)
FEATURE_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_feature_columns.json"
)
AUG_CONFIG_PATH = (
    ROOT / "configs" / "day09_augmentation.json"
)
DEFAULT_OUTPUT_PATH = (
    ROOT
    / "data"
    / "day09"
    / "sequences_augmented"
    / "train.csv"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--jitter-ratio",
        type=float,
        default=None,
        help=(
            "생략하면 day09_augmentation.json의 "
            "jitter_ratio를 사용합니다."
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
    )

    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def make_sequence_row(
    item,
    frame_df,
    augmentation_name: str,
    target_steps: int,
    min_valid_frames: int,
    feature_names: list[str],
):
    frame_matrix, frame_ids = build_frame_feature_matrix(
        frame_df
    )

    if len(frame_matrix) < min_valid_frames:
        return None

    sequence = resample_sequence(
        frame_matrix,
        frame_ids,
        target_steps=target_steps,
    )
    flattened = flatten_sequence(sequence)

    if len(flattened) != len(feature_names):
        raise RuntimeError(
            "보강 Sequence Feature 수가 "
            "6일차 Feature 수와 다릅니다."
        )

    row = {
        "clip_id": (
            f"{item['clip_id']}__{augmentation_name}"
        ),
        "source_clip_id": item["clip_id"],
        "subject": item["subject"],
        "action_id": item["action_id"],
        "action_label": item["action_label"],
        "take": item["take"],
        "augmentation": augmentation_name,
        "valid_frames": len(frame_matrix),
        "first_valid_frame": int(frame_ids[0]),
        "last_valid_frame": int(frame_ids[-1]),
    }

    for name, value in zip(feature_names, flattened):
        row[name] = float(value)

    return row


def main() -> None:
    args = parse_args()

    training_info = load_json(TRAIN_INFO_PATH)
    feature_columns = load_json(FEATURE_PATH)
    aug_config = load_json(AUG_CONFIG_PATH)

    target_steps = int(training_info["sequence_steps"])
    min_valid_frames = int(
        training_info["min_valid_frames"]
    )

    expected_columns = sequence_feature_names(
        target_steps
    )

    if feature_columns != expected_columns:
        raise RuntimeError(
            "6일차 Feature Column과 현재 코드가 다릅니다."
        )

    train = pd.read_csv(TRAIN_SPLIT)

    if len(train) != 6:
        raise RuntimeError(
            "현재 기본 Train은 S01의 6개 Clip입니다."
        )

    jitter_ratio = (
        float(args.jitter_ratio)
        if args.jitter_ratio is not None
        else float(
            aug_config["jitter_ratio"]
        )
    )

    if jitter_ratio < 0:
        raise ValueError(
            "jitter_ratio는 0 이상이어야 합니다."
        )

    output_path = Path(
        args.output
    )

    if not output_path.is_absolute():
        output_path = (
            ROOT
            / output_path
        )

    seed = int(aug_config["random_state"])
    rows = []

    for index, item in train.iterrows():
        keypoint_path = ROOT / str(item["keypoint_csv"])
        original_df = pd.read_csv(keypoint_path)

        original_row = make_sequence_row(
            item,
            original_df,
            "original",
            target_steps,
            min_valid_frames,
            feature_columns,
        )

        flip_df = horizontal_flip_dataframe(original_df)
        flip_row = make_sequence_row(
            item,
            flip_df,
            "flip",
            target_steps,
            min_valid_frames,
            feature_columns,
        )

        jitter_df = jitter_dataframe(
            original_df,
            jitter_ratio=jitter_ratio,
            random_state=seed + int(index),
        )
        jitter_row = make_sequence_row(
            item,
            jitter_df,
            f"jitter_{jitter_ratio:.3f}",
            target_steps,
            min_valid_frames,
            feature_columns,
        )

        for row in [original_row, flip_row, jitter_row]:
            if row is None:
                raise RuntimeError(
                    f"보강 Sequence 생성 실패: "
                    f"{item['clip_id']}"
                )
            rows.append(row)

        print(
            f"{item['clip_id']} "
            "→ original / flip / jitter"
        )

    result = pd.DataFrame(rows)

    if len(result) != 18:
        raise RuntimeError(
            f"Augmented Train은 18행이어야 합니다: "
            f"{len(result)}"
        )

    if result["action_label"].nunique() != 6:
        raise RuntimeError(
            "Augmented Train에 6개 Action이 모두 필요합니다."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(f"Sequence steps : {target_steps}")
    print(f"Original clips : {len(train)}")
    print(f"Augmented rows : {len(result)}")
    print()
    print(result["augmentation"].value_counts())
    print()
    print(f"Jitter ratio   : {jitter_ratio}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()