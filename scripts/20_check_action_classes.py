from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ACTION_FILE = (
    ROOT
    / "configs"
    / "action_classes.csv"
)

VIDEO_DIR = (
    ROOT
    / "data"
    / "day04"
    / "videos"
)


EXPECTED_ACTIONS = {
    "A001": (
        "bend_return",
        "bend_return.mp4",
    ),
    "A002": (
        "leg_raise_lower",
        "leg_raise_lower.mp4",
    ),
    "A003": (
        "walk_turn_walk",
        "walk_turn_walk.mp4",
    ),
    "A004": (
        "drink_return",
        "drink_return.mp4",
    ),
    "A005": (
        "sit_stand",
        "sit_stand.mp4",
    ),
    "A006": (
        "wave",
        "wave.mp4",
    ),
}


def find_duplicates(
    values: list[str],
) -> set[str]:
    return {
        value
        for value in values
        if values.count(value) > 1
    }


def main() -> None:
    if not ACTION_FILE.exists():
        raise FileNotFoundError(
            ACTION_FILE
        )

    with ACTION_FILE.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "action_id",
            "action_label",
            "type",
            "video_file",
            "description",
        }

        actual_columns = set(
            reader.fieldnames or []
        )

        missing_columns = (
            required_columns
            - actual_columns
        )

        if missing_columns:
            raise RuntimeError(
                "필수 열이 없습니다: "
                + ", ".join(
                    sorted(missing_columns)
                )
            )

        rows = list(reader)

    if not rows:
        raise RuntimeError(
            "Action 정의가 없습니다."
        )

    problems = []

    if len(rows) != len(
        EXPECTED_ACTIONS
    ):
        problems.append(
            "Action 개수는 6개여야 합니다. "
            f"현재 {len(rows)}개입니다."
        )

    ids = [
        row["action_id"].strip()
        for row in rows
    ]

    labels = [
        row["action_label"].strip()
        for row in rows
    ]

    files = [
        row["video_file"].strip()
        for row in rows
    ]

    duplicate_ids = (
        find_duplicates(ids)
    )

    duplicate_labels = (
        find_duplicates(labels)
    )

    duplicate_files = (
        find_duplicates(files)
    )

    if duplicate_ids:
        problems.append(
            "중복 Action ID: "
            + ", ".join(
                sorted(duplicate_ids)
            )
        )

    if duplicate_labels:
        problems.append(
            "중복 Action Label: "
            + ", ".join(
                sorted(
                    duplicate_labels
                )
            )
        )

    if duplicate_files:
        problems.append(
            "중복 video_file: "
            + ", ".join(
                sorted(duplicate_files)
            )
        )

    print(
        f"Action count: {len(rows)}"
    )

    print()

    for row in rows:
        action_id = (
            row["action_id"]
            .strip()
        )

        action_label = (
            row["action_label"]
            .strip()
        )

        video_file = (
            row["video_file"]
            .strip()
        )

        action_type = (
            row["type"]
            .strip()
        )

        video_path = (
            VIDEO_DIR
            / video_file
        )

        exists = (
            video_path.exists()
        )

        print(
            f"{action_id} "
            f"{action_label:18s} "
            f"file="
            f"{video_file:24s} "
            f"exists={exists}"
        )

        expected = (
            EXPECTED_ACTIONS.get(
                action_id
            )
        )

        if expected is None:
            problems.append(
                "알 수 없는 Action ID: "
                f"{action_id}"
            )
        else:
            expected_label, (
                expected_file
            ) = expected

            if action_label != (
                expected_label
            ):
                problems.append(
                    f"{action_id}의 Label은 "
                    f"{expected_label}이어야 합니다."
                )

            if video_file != (
                expected_file
            ):
                problems.append(
                    f"{action_id}의 video_file은 "
                    f"{expected_file}이어야 합니다."
                )

        if action_type != "dynamic":
            problems.append(
                f"{action_id}의 type은 "
                "dynamic이어야 합니다."
            )

        if not exists:
            problems.append(
                "없는 동영상: "
                f"{video_file}"
            )

    expected_ids = set(
        EXPECTED_ACTIONS.keys()
    )

    missing_ids = (
        expected_ids
        - set(ids)
    )

    if missing_ids:
        problems.append(
            "누락 Action ID: "
            + ", ".join(
                sorted(missing_ids)
            )
        )

    print()

    if problems:
        for problem in dict.fromkeys(
            problems
        ):
            print(
                "[ERROR]",
                problem,
            )

        raise RuntimeError(
            "Action 설정을 "
            "다시 확인하세요."
        )

    print(
        "Action class check: OK"
    )


if __name__ == "__main__":
    main()