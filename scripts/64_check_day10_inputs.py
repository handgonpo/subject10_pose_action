from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.sequence_utils import (
    sequence_feature_names,
)


MODEL_PATH = (
    ROOT / "artifacts" / "day06"
    / "rf_action_baseline.joblib"
)

FEATURE_PATH = (
    ROOT / "artifacts" / "day06"
    / "rf_feature_columns.json"
)

TRAIN_INFO_PATH = (
    ROOT / "artifacts" / "day06"
    / "rf_training_info.json"
)

SPLIT_DIR = (
    ROOT / "data" / "day05" / "splits"
)

OUTPUT_PATH = (
    ROOT / "reports" / "day10"
    / "tables" / "day10_input_check.csv"
)

EXPECTED_ACTIONS = {
    "bend_return",
    "leg_raise_lower",
    "walk_turn_walk",
    "drink_return",
    "sit_stand",
    "wave",
}

EXPECTED_SPLITS = {
    "train": "S01",
    "val": "S02",
    "test": "S03",
}

CORE_FILES = [
    MODEL_PATH,
    FEATURE_PATH,
    TRAIN_INFO_PATH,
    SPLIT_DIR / "train.csv",
    SPLIT_DIR / "val.csv",
    SPLIT_DIR / "test.csv",
    ROOT / "configs" / "day07_realtime.json",
    ROOT / "configs" / "day08_multi_person.json",
    ROOT / "configs" / "day09_augmentation.json",
    ROOT / "configs" / "day09_vae.json",
]

EVIDENCE_FILES = [
    ROOT / "data" / "day05" / "qa"
    / "keypoint_clip_review.csv",

    ROOT / "data" / "day05" / "manifests"
    / "approved_clip_manifest.csv",

    ROOT / "reports" / "day06" / "metrics"
    / "validation_predictions.csv",

    ROOT / "reports" / "day06" / "misclassified"
    / "validation_misclassified.csv",

    ROOT / "reports" / "day07"
    / "realtime_run_summary.csv",

    ROOT / "reports" / "day08" / "experiments"
    / "tracking_scenario_summary.csv",

    ROOT / "reports" / "day09" / "augmentation"
    / "baseline_vs_augmented.csv",

    ROOT / "reports" / "day09" / "vae"
    / "action_error_summary.csv",
]

REQUIRED_TRAIN_INFO_KEYS = {
    "classes",
    "feature_count",
    "sequence_steps",
}


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main() -> None:
    rows = []
    missing_core = []

    for path in CORE_FILES:
        exists = path.exists()

        rows.append(
            {
                "type": "CORE",
                "path": str(path.relative_to(ROOT)),
                "exists": int(exists),
            }
        )

        if not exists:
            missing_core.append(path)

    for path in EVIDENCE_FILES:
        rows.append(
            {
                "type": "EVIDENCE",
                "path": str(path.relative_to(ROOT)),
                "exists": int(path.exists()),
            }
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
            fieldnames=["type", "path", "exists"],
        )
        writer.writeheader()
        writer.writerows(rows)

    if missing_core:
        print("[ERROR] 필수 파일이 없습니다.")

        for path in missing_core:
            print("-", path.relative_to(ROOT))

        raise RuntimeError(
            "10일차 핵심 입력을 먼저 복구하세요."
        )

    training_info = load_json(TRAIN_INFO_PATH)

    missing_info = (
        REQUIRED_TRAIN_INFO_KEYS - set(training_info)
    )

    if missing_info:
        raise RuntimeError(
            "rf_training_info.json에 필수 정보가 없습니다: "
            + ", ".join(sorted(missing_info))
        )

    feature_columns = load_json(FEATURE_PATH)
    model = joblib.load(MODEL_PATH)

    sequence_steps = int(
        training_info["sequence_steps"]
    )
    expected_feature_count = sequence_steps * 38

    if len(feature_columns) != expected_feature_count:
        raise RuntimeError(
            "Feature Column 수가 "
            "sequence_steps × 38과 다릅니다."
        )

    expected_columns = sequence_feature_names(
        sequence_steps
    )

    if feature_columns != expected_columns:
        raise RuntimeError(
            "6일차 Feature Column의 이름 또는 순서가 "
            "현재 sequence_utils.py와 다릅니다."
        )

    if int(training_info["feature_count"]) != (
        expected_feature_count
    ):
        raise RuntimeError(
            "Training Info의 feature_count가 "
            "sequence_steps × 38과 다릅니다."
        )

    if int(
        getattr(model, "n_features_in_", -1)
    ) != expected_feature_count:
        raise RuntimeError(
            "Random Forest 입력 Feature 수가 "
            "현재 Training Info와 다릅니다."
        )

    model_classes = set(map(str, model.classes_))
    info_classes = set(
        map(str, training_info["classes"])
    )

    if (
        model_classes != EXPECTED_ACTIONS
        or info_classes != EXPECTED_ACTIONS
    ):
        raise RuntimeError(
            "6일차 Action Class가 "
            "현재 6개 Action과 다릅니다."
        )

    for split_name, subject in (
        EXPECTED_SPLITS.items()
    ):
        path = SPLIT_DIR / f"{split_name}.csv"
        df = pd.read_csv(path)

        if len(df) != 6:
            raise RuntimeError(
                f"{split_name}은 6개 Clip이어야 합니다."
            )

        if set(df["subject"].astype(str)) != {
            subject
        }:
            raise RuntimeError(
                f"{split_name} Subject가 "
                f"{subject}가 아닙니다."
            )

        if set(
            df["action_label"].astype(str)
        ) != EXPECTED_ACTIONS:
            raise RuntimeError(
                f"{split_name} Action Coverage를 "
                "확인하세요."
            )

    evidence_ready = sum(
        int(path.exists())
        for path in EVIDENCE_FILES
    )

    print(f"Sequence steps : {sequence_steps}")
    print(
        f"Feature count  : {expected_feature_count}"
    )
    print("Action classes : OK")
    print(
        "Subject split  : S01 / S02 / S03"
    )
    print(
        "Evidence files : "
        f"{evidence_ready} / "
        f"{len(EVIDENCE_FILES)}"
    )
    print()

    missing_evidence = [
        path
        for path in EVIDENCE_FILES
        if not path.exists()
    ]

    if missing_evidence:
        print(
            "[WARNING] 일부 이전 분석 결과가 없습니다."
        )
        print(
            "Failure Catalog에는 존재하는 결과만 "
            "사용됩니다."
        )

        for path in missing_evidence:
            print("-", path.relative_to(ROOT))

        print()

    print(f"Saved: {OUTPUT_PATH}")
    print("Day10 core inputs: READY")


if __name__ == "__main__":
    main()