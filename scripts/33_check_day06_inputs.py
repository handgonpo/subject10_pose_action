from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

APPROVED_PATH = (
    ROOT
    / "data"
    / "day05"
    / "manifests"
    / "approved_clip_manifest.csv"
)

SPLIT_DIR = (
    ROOT
    / "data"
    / "day05"
    / "splits"
)

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day06"
    / "tables"
    / "day06_input_check.csv"
)

EXPECTED = {
    "train": {
        "subject": "S01",
        "count": 6,
    },
    "val": {
        "subject": "S02",
        "count": 6,
    },
    "test": {
        "subject": "S03",
        "count": 6,
    },
}

EXPECTED_ACTIONS = {
    "A001",
    "A002",
    "A003",
    "A004",
    "A005",
    "A006",
}


def main() -> None:
    if not APPROVED_PATH.exists():
        raise FileNotFoundError(
            APPROVED_PATH
        )

    approved = pd.read_csv(
        APPROVED_PATH
    )

    if len(approved) != 18:
        raise RuntimeError(
            "현재 6일차 기본 실습은 "
            "18개 PASS Clip을 사용합니다. "
            f"Approved Clip: {len(approved)}"
        )

    split_frames = {}
    rows = []
    problems = []

    for split_name, rule in (
        EXPECTED.items()
    ):
        split_path = (
            SPLIT_DIR
            / f"{split_name}.csv"
        )

        if not split_path.exists():
            raise FileNotFoundError(
                split_path
            )

        df = pd.read_csv(
            split_path
        )

        split_frames[
            split_name
        ] = df

        if len(df) != rule["count"]:
            problems.append(
                f"{split_name}: "
                f"Clip 수 {len(df)}"
            )

        subjects = set(
            df["subject"]
            .astype(str)
        )

        if subjects != {
            rule["subject"]
        }:
            problems.append(
                f"{split_name}: "
                f"Subject {sorted(subjects)}"
            )

        actions = set(
            df["action_id"]
            .astype(str)
            .str.upper()
        )

        if actions != EXPECTED_ACTIONS:
            problems.append(
                f"{split_name}: "
                "Action Coverage 오류"
            )

        for _, item in (
            df.iterrows()
        ):
            keypoint_path = (
                ROOT
                / str(
                    item[
                        "keypoint_csv"
                    ]
                )
            )

            exists = (
                keypoint_path.exists()
            )

            row_count = 0

            if exists:
                keypoint_df = (
                    pd.read_csv(
                        keypoint_path
                    )
                )

                row_count = len(
                    keypoint_df
                )

            if (
                not exists
                or row_count == 0
            ):
                problems.append(
                    "Keypoint CSV 오류: "
                    f"{item['clip_id']}"
                )

            rows.append(
                {
                    "split": (
                        split_name
                    ),
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
                    "keypoint_csv": (
                        item[
                            "keypoint_csv"
                        ]
                    ),
                    "exists": int(
                        exists
                    ),
                    "rows": (
                        row_count
                    ),
                }
            )

    clip_sets = {
        name: set(
            df["clip_id"]
            .astype(str)
        )
        for name, df in (
            split_frames.items()
        )
    }

    pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]

    for left, right in pairs:
        overlap = (
            clip_sets[left]
            & clip_sets[right]
        )

        if overlap:
            problems.append(
                f"{left}/{right} "
                "Clip Leakage: "
                f"{sorted(overlap)}"
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
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

    if problems:
        for problem in problems:
            print(
                "[ERROR]",
                problem,
            )

        raise RuntimeError(
            "6일차 입력 데이터에 "
            "문제가 있습니다."
        )

    print(
        "Train      : S01 / 6 Clips"
    )
    print(
        "Validation : S02 / 6 Clips"
    )
    print(
        "Test       : S03 / 6 Clips"
    )
    print()
    print(
        "All Keypoint CSV: OK"
    )
    print(
        "Action Coverage  : OK"
    )
    print(
        "Clip Leakage     : NONE"
    )
    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print(
        "Day06 input data: READY"
    )


if __name__ == "__main__":
    main()