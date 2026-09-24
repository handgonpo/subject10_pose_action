from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(ROOT),
)


from src.sequence_utils import (
    build_frame_feature_matrix,
    flatten_sequence,
    resample_sequence,
    sequence_feature_names,
)


CONFIG_PATH = (
    ROOT
    / "configs"
    / "day06_baseline.json"
)

SPLIT_DIR = (
    ROOT
    / "data"
    / "day05"
    / "splits"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "day06"
    / "sequences"
)

EXPECTED_ACTIONS = {
    "A001",
    "A002",
    "A003",
    "A004",
    "A005",
    "A006",
}


def load_day06_config():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def build_split_dataset(
    split_name: str,
    config: dict,
) -> Path:
    split_path = (
        SPLIT_DIR
        / f"{split_name}.csv"
    )

    if not split_path.exists():
        raise FileNotFoundError(
            split_path
        )

    manifest = pd.read_csv(
        split_path
    )

    target_steps = int(
        config["sequence_steps"]
    )

    min_valid_frames = int(
        config[
            "min_valid_frames"
        ]
    )

    feature_names = (
        sequence_feature_names(
            target_steps
        )
    )

    rows = []
    skipped = []

    for _, item in (
        manifest.iterrows()
    ):
        keypoint_path = (
            ROOT
            / str(
                item[
                    "keypoint_csv"
                ]
            )
        )

        if not keypoint_path.exists():
            skipped.append(
                {
                    "clip_id": (
                        item["clip_id"]
                    ),
                    "reason": (
                        "KEYPOINT_CSV_MISSING"
                    ),
                }
            )
            continue

        frame_df = pd.read_csv(
            keypoint_path
        )

        (
            frame_matrix,
            frame_ids,
        ) = (
            build_frame_feature_matrix(
                frame_df
            )
        )

        if (
            len(frame_matrix)
            < min_valid_frames
        ):
            skipped.append(
                {
                    "clip_id": (
                        item["clip_id"]
                    ),
                    "reason": (
                        "TOO_FEW_VALID_FRAMES"
                    ),
                }
            )
            continue

        sequence = (
            resample_sequence(
                frame_matrix,
                frame_ids,
                target_steps=(
                    target_steps
                ),
            )
        )

        flattened = (
            flatten_sequence(
                sequence
            )
        )

        if len(flattened) != (
            target_steps * 38
        ):
            raise RuntimeError(
                "Sequence Feature 수가 "
                "예상과 다릅니다."
            )

        row = {
            "clip_id": (
                item["clip_id"]
            ),
            "subject": (
                item["subject"]
            ),
            "action_id": (
                item["action_id"]
            ),
            "action_label": (
                item[
                    "action_label"
                ]
            ),
            "take": item["take"],
            "valid_frames": (
                len(frame_matrix)
            ),
            "first_valid_frame": (
                int(frame_ids[0])
            ),
            "last_valid_frame": (
                int(frame_ids[-1])
            ),
        }

        for name, value in zip(
            feature_names,
            flattened,
        ):
            row[name] = float(value)

        rows.append(row)

        print(
            f"{split_name:5s} "
            f"{item['clip_id']} "
            f"valid={len(frame_matrix)} "
            f"→ {target_steps} steps"
        )

    if skipped:
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        skip_path = (
            OUTPUT_DIR
            / f"{split_name}_skipped.csv"
        )

        pd.DataFrame(
            skipped
        ).to_csv(
            skip_path,
            index=False,
            encoding="utf-8-sig",
        )

    result = pd.DataFrame(rows)

    if len(result) != 6:
        raise RuntimeError(
            f"{split_name} Sequence는 "
            "6개가 모두 생성되어야 합니다. "
            f"현재 {len(result)}개입니다."
        )

    actions = set(
        result["action_id"]
        .astype(str)
        .str.upper()
    )

    if actions != EXPECTED_ACTIONS:
        raise RuntimeError(
            f"{split_name}에 "
            "A001~A006이 모두 없습니다."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{split_name}.csv"
    )

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"{split_name} Sequence: "
        f"{len(result)}"
    )
    print(
        f"Saved: {output_path}"
    )
    print()

    return output_path


def main() -> None:
    config = load_day06_config()

    for split_name in [
        "train",
        "val",
        "test",
    ]:
        build_split_dataset(
            split_name,
            config,
        )

    print(
        "Day06 Sequence Dataset: READY"
    )


if __name__ == "__main__":
    main()