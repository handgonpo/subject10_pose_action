
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST_PATH = (
    ROOT
    / "data"
    / "day05"
    / "manifests"
    / "approved_clip_manifest.csv"
)

DEFAULT_SPLIT_CONFIG = (
    ROOT
    / "configs"
    / "day05_subject_split.json"
)

DEFAULT_OUTPUT_DIR = (
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
        "--manifest-path",
        default=str(
            DEFAULT_MANIFEST_PATH
        ),
    )

    parser.add_argument(
        "--split-config",
        default=str(
            DEFAULT_SPLIT_CONFIG
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--expected-actions",
        default=DEFAULT_EXPECTED_ACTIONS,
        help=(
            "쉼표로 구분합니다. "
            "예: A006"
        ),
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


def load_split_config(
    split_config: Path,
) -> dict:
    with split_config.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_subject_sets(
    train_subjects,
    val_subjects,
    test_subjects,
) -> None:
    train = set(train_subjects)
    val = set(val_subjects)
    test = set(test_subjects)

    if train & val:
        raise RuntimeError(
            "Train / Validation "
            "Subject 중복"
        )

    if train & test:
        raise RuntimeError(
            "Train / Test "
            "Subject 중복"
        )

    if val & test:
        raise RuntimeError(
            "Validation / Test "
            "Subject 중복"
        )


def require_action_coverage(
    split_name: str,
    df: pd.DataFrame,
    expected_actions: set[str],
) -> None:
    if df.empty:
        raise RuntimeError(
            f"{split_name} 데이터가 "
            "비어 있습니다."
        )

    actual_actions = set(
        df["action_id"]
        .astype(str)
        .str.upper()
    )

    missing_actions = (
        expected_actions
        - actual_actions
    )

    if missing_actions:
        raise RuntimeError(
            f"{split_name}에 없는 Action: "
            + ", ".join(
                sorted(
                    missing_actions
                )
            )
        )

    print()
    print(f"[{split_name}]")
    print(
        "Subjects:",
        sorted(
            df["subject"].unique()
        ),
    )
    print("Clips:", len(df))
    print(
        df["action_label"]
        .value_counts()
        .sort_index()
    )


def main() -> None:
    args = parse_args()

    manifest_path = Path(
        args.manifest_path
    )
    split_config = Path(
        args.split_config
    )
    output_dir = Path(
        args.output_dir
    )

    expected_actions = (
        parse_expected_actions(
            args.expected_actions
        )
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            manifest_path
        )

    if not split_config.exists():
        raise FileNotFoundError(
            split_config
        )

    config = load_split_config(
        split_config
    )

    train_subjects = (
        config["train_subjects"]
    )
    val_subjects = (
        config["val_subjects"]
    )
    test_subjects = (
        config["test_subjects"]
    )

    validate_subject_sets(
        train_subjects,
        val_subjects,
        test_subjects,
    )

    df = pd.read_csv(
        manifest_path
    )

    configured_subjects = set(
        train_subjects
        + val_subjects
        + test_subjects
    )

    data_subjects = set(
        df["subject"].astype(str)
    )

    unassigned = (
        data_subjects
        - configured_subjects
    )

    missing_from_data = (
        configured_subjects
        - data_subjects
    )

    if unassigned:
        raise RuntimeError(
            "Split에 배정되지 않은 Subject: "
            + ", ".join(
                sorted(unassigned)
            )
        )

    if missing_from_data:
        raise RuntimeError(
            "Approved 데이터에 없는 "
            "Subject가 설정에 있습니다: "
            + ", ".join(
                sorted(
                    missing_from_data
                )
            )
        )

    train_df = df[
        df["subject"].isin(
            train_subjects
        )
    ].copy()

    val_df = df[
        df["subject"].isin(
            val_subjects
        )
    ].copy()

    test_df = df[
        df["subject"].isin(
            test_subjects
        )
    ].copy()

    require_action_coverage(
        "TRAIN",
        train_df,
        expected_actions,
    )
    require_action_coverage(
        "VALIDATION",
        val_df,
        expected_actions,
    )
    require_action_coverage(
        "TEST",
        test_df,
        expected_actions,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    for name, split_df in (
        outputs.items()
    ):
        path = (
            output_dir
            / f"{name}.csv"
        )

        split_df.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"Saved: {path}")

    print()
    print(
        "Subject-wise split: COMPLETE"
    )


if __name__ == "__main__":
    main()