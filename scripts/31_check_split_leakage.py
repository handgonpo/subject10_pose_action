
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SPLIT_DIR = (
    ROOT
    / "data"
    / "day05"
    / "splits"
)

DEFAULT_EXPECTED_ACTIONS = (
    "A001,A002,A003,A004,A005,A006"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split-dir",
        default=str(
            DEFAULT_SPLIT_DIR
        ),
    )

    parser.add_argument(
        "--expected-actions",
        default=DEFAULT_EXPECTED_ACTIONS,
    )

    return parser.parse_args()


def parse_expected_actions(
    value: str,
) -> set[str]:
    actions = {
        item.strip().upper()
        for item in value.split(",")
        if item.strip()
    }

    if not actions:
        raise ValueError(
            "--expected-actions가 "
            "비어 있습니다."
        )

    return actions


def load_split(
    split_dir: Path,
    name: str,
) -> pd.DataFrame:
    path = (
        split_dir
        / f"{name}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def main() -> None:
    args = parse_args()

    split_dir = Path(
        args.split_dir
    )

    expected_actions = (
        parse_expected_actions(
            args.expected_actions
        )
    )

    train = load_split(
        split_dir,
        "train",
    )
    val = load_split(
        split_dir,
        "val",
    )
    test = load_split(
        split_dir,
        "test",
    )

    datasets = {
        "TRAIN": train,
        "VALIDATION": val,
        "TEST": test,
    }

    errors = []

    for name, df in (
        datasets.items()
    ):
        actual_actions = set(
            df["action_id"]
            .astype(str)
            .str.upper()
        )

        missing = (
            expected_actions
            - actual_actions
        )

        if missing:
            errors.append(
                f"{name} Action 누락: "
                + ", ".join(
                    sorted(missing)
                )
            )

    train_subjects = set(
        train["subject"].astype(str)
    )
    val_subjects = set(
        val["subject"].astype(str)
    )
    test_subjects = set(
        test["subject"].astype(str)
    )

    subject_pairs = [
        (
            "Train/Validation",
            train_subjects
            & val_subjects,
        ),
        (
            "Train/Test",
            train_subjects
            & test_subjects,
        ),
        (
            "Validation/Test",
            val_subjects
            & test_subjects,
        ),
    ]

    for label, overlap in (
        subject_pairs
    ):
        if overlap:
            errors.append(
                f"{label} Subject Leakage: "
                f"{sorted(overlap)}"
            )

    train_clips = set(
        train["clip_id"].astype(str)
    )
    val_clips = set(
        val["clip_id"].astype(str)
    )
    test_clips = set(
        test["clip_id"].astype(str)
    )

    clip_pairs = [
        (
            "Train/Validation",
            train_clips
            & val_clips,
        ),
        (
            "Train/Test",
            train_clips
            & test_clips,
        ),
        (
            "Validation/Test",
            val_clips
            & test_clips,
        ),
    ]

    for label, overlap in clip_pairs:
        if overlap:
            errors.append(
                f"{label} Clip Leakage: "
                f"{sorted(overlap)}"
            )

    print(
        "Train Subjects:",
        sorted(train_subjects),
    )
    print(
        "Validation Subjects:",
        sorted(val_subjects),
    )
    print(
        "Test Subjects:",
        sorted(test_subjects),
    )
    print()

    if errors:
        for error in errors:
            print("[FAIL]", error)

        raise RuntimeError(
            "Split 검사에서 문제가 "
            "발견되었습니다."
        )

    print(
        "[PASS] Subject overlap 없음"
    )
    print(
        "[PASS] Clip overlap 없음"
    )
    print(
        "[PASS] 필요한 Action Coverage 확인"
    )
    print()
    print("Leakage check: OK")


if __name__ == "__main__":
    main()