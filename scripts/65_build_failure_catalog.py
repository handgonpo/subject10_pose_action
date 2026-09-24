from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    ROOT / "reports" / "day10"
    / "failures" / "failure_catalog.csv"
)

DAY05_REVIEW = (
    ROOT / "data" / "day05" / "qa"
    / "keypoint_clip_review.csv"
)

DAY06_PREDICTIONS = (
    ROOT / "reports" / "day06" / "metrics"
    / "validation_predictions.csv"
)

DAY07_SUMMARY = (
    ROOT / "reports" / "day07"
    / "realtime_run_summary.csv"
)

DAY07_CONFIG = (
    ROOT / "configs" / "day07_realtime.json"
)

DAY08_SUMMARY = (
    ROOT / "reports" / "day08" / "experiments"
    / "tracking_scenario_summary.csv"
)

DAY09_AUGMENTATION = (
    ROOT / "reports" / "day09" / "augmentation"
    / "baseline_vs_augmented.csv"
)

DAY09_VAE = (
    ROOT / "reports" / "day09" / "vae"
    / "action_error_summary.csv"
)


def safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def load_runtime_confidence_threshold() -> float:
    if not DAY07_CONFIG.exists():
        raise FileNotFoundError(DAY07_CONFIG)

    with DAY07_CONFIG.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    if "confidence_threshold" not in config:
        raise RuntimeError(
            "day07_realtime.json에 "
            "confidence_threshold가 없습니다."
        )

    return float(
        config["confidence_threshold"]
    )


def add_case(
    rows: list[dict],
    *,
    case_id: str,
    source_day: str,
    category: str,
    severity: str,
    symptom: str,
    evidence: str,
    failure_candidate: bool,
) -> None:
    rows.append(
        {
            "case_id": case_id,
            "source_day": source_day,
            "category": category,
            "severity": severity,
            "symptom": symptom,
            "evidence": evidence,
            "failure_candidate": int(
                failure_candidate
            ),
            "student_selected": "",
            "root_cause_stage": "",
            "hypothesis": "",
            "changed_variable": "",
            "result": "",
        }
    )


def add_day05(rows) -> None:
    if not DAY05_REVIEW.exists():
        return

    df = pd.read_csv(DAY05_REVIEW)

    for _, item in df.iterrows():
        qa_status = safe_text(
            item.get("qa_status", "")
        ).upper()

        decision = safe_text(
            item.get("human_decision", "")
        ).upper()

        is_candidate = (
            qa_status == "CHECK"
            or decision in {
                "RESHOOT",
                "EXCLUDE",
            }
        )

        if not is_candidate:
            continue

        clip_id = safe_text(
            item.get("clip_id", "UNKNOWN")
        )

        evidence = (
            f"qa={qa_status}; "
            f"decision={decision}; "
            f"pose_ratio="
            f"{safe_text(item.get('pose_ratio', ''))}; "
            f"mean_conf="
            f"{safe_text(item.get('mean_keypoint_conf', ''))}; "
            f"note="
            f"{safe_text(item.get('human_note', ''))}"
        )

        add_case(
            rows,
            case_id=f"D05_{clip_id}",
            source_day="DAY05",
            category="POSE_OR_DATA",
            severity="REVIEW",
            symptom=(
                "5일차 QA에서 "
                "재검토가 필요한 Clip"
            ),
            evidence=evidence,
            failure_candidate=True,
        )


def add_day06(
    rows,
    confidence_threshold: float,
) -> None:
    if not DAY06_PREDICTIONS.exists():
        return

    df = pd.read_csv(DAY06_PREDICTIONS)

    if df.empty:
        return

    for _, item in df.iterrows():
        correct_value = item.get(
            "correct",
            False,
        )

        correct = str(
            correct_value
        ).strip().lower() in {
            "true",
            "1",
            "yes",
        }

        confidence = safe_float(
            item.get("confidence", None),
            default=0.0,
        )

        is_candidate = (
            not correct
            or confidence
            < confidence_threshold
        )

        if not is_candidate:
            continue

        clip_id = safe_text(
            item.get("clip_id", "UNKNOWN")
        )

        add_case(
            rows,
            case_id=f"D06_{clip_id}",
            source_day="DAY06",
            category="ACTION",
            severity=(
                "HIGH"
                if not correct
                else "REVIEW"
            ),
            symptom=(
                "Validation 오분류"
                if not correct
                else (
                    "Validation 정답이지만 "
                    "Confidence가 낮음"
                )
            ),
            evidence=(
                f"true="
                f"{safe_text(item.get('true_label', ''))}; "
                f"pred="
                f"{safe_text(item.get('pred_label', ''))}; "
                f"confidence="
                f"{confidence:.4f}"
            ),
            failure_candidate=True,
        )


def add_day07(rows) -> None:
    if not DAY07_SUMMARY.exists():
        return

    df = pd.read_csv(DAY07_SUMMARY)

    for _, item in df.iterrows():
        unknown_ratio = safe_float(
            item.get("unknown_ratio", 0.0),
            default=0.0,
        )

        stable_match = safe_float(
            item.get("stable_match_ratio", None),
            default=None,
        )

        changes = int(
            safe_float(
                item.get(
                    "stable_label_changes",
                    0,
                ),
                default=0,
            )
        )

        is_candidate = (
            unknown_ratio > 0.30
            or (
                stable_match is not None
                and stable_match < 0.70
            )
            or changes >= 3
        )

        if not is_candidate:
            continue

        run = safe_text(
            item.get("run", "UNKNOWN")
        )

        stable_text = (
            ""
            if stable_match is None
            else f"{stable_match:.4f}"
        )

        add_case(
            rows,
            case_id=f"D07_{run}",
            source_day="DAY07",
            category="REALTIME",
            severity="REVIEW",
            symptom=(
                "Realtime 안정성 또는 "
                "UNKNOWN 문제"
            ),
            evidence=(
                f"unknown_ratio="
                f"{unknown_ratio:.4f}; "
                f"stable_match_ratio="
                f"{stable_text}; "
                f"stable_changes="
                f"{changes}; "
                f"fps="
                f"{safe_text(item.get('mean_processing_fps', ''))}"
            ),
            failure_candidate=True,
        )


def add_day08(rows) -> None:
    if not DAY08_SUMMARY.exists():
        return

    df = pd.read_csv(DAY08_SUMMARY)

    for _, item in df.iterrows():
        extra_ids = int(
            safe_float(
                item.get("extra_id_count", 0),
                default=0,
            )
        )

        scenario = safe_text(
            item.get("scenario", "UNKNOWN")
        )

        two_ratio = safe_float(
            item.get("two_id_frame_ratio", 0.0),
            default=0.0,
        )

        add_case(
            rows,
            case_id=f"D08_{scenario}",
            source_day="DAY08",
            category="TRACKING",
            severity=(
                "REVIEW"
                if extra_ids > 0
                else "OBSERVE"
            ),
            symptom=(
                "추가 Track ID 발생 가능성"
                if extra_ids > 0
                else "Tracking Scenario 관찰"
            ),
            evidence=(
                f"unique_track_ids="
                f"{safe_text(item.get('unique_track_ids', ''))}; "
                f"two_id_frame_ratio="
                f"{two_ratio:.4f}; "
                f"extra_id_count="
                f"{extra_ids}"
            ),
            failure_candidate=(
                extra_ids > 0
            ),
        )


def add_day09_augmentation(rows) -> None:
    if not DAY09_AUGMENTATION.exists():
        return

    df = pd.read_csv(DAY09_AUGMENTATION)

    if df.empty or "model" not in df.columns:
        return

    indexed = {
        str(row["model"]): row
        for _, row in df.iterrows()
    }

    if (
        "baseline" not in indexed
        or "augmented" not in indexed
    ):
        return

    baseline_f1 = safe_float(
        indexed["baseline"].get(
            "macro_f1",
            None,
        )
    )
    augmented_f1 = safe_float(
        indexed["augmented"].get(
            "macro_f1",
            None,
        )
    )

    if (
        baseline_f1 is None
        or augmented_f1 is None
    ):
        return

    degraded = augmented_f1 < baseline_f1

    add_case(
        rows,
        case_id="D09_AUGMENTATION",
        source_day="DAY09",
        category="DATA",
        severity=(
            "REVIEW"
            if degraded
            else "OBSERVE"
        ),
        symptom=(
            "보강 후 Macro F1 하락"
            if degraded
            else "보강 전후 성능 비교"
        ),
        evidence=(
            f"baseline_macro_f1="
            f"{baseline_f1:.4f}; "
            f"augmented_macro_f1="
            f"{augmented_f1:.4f}"
        ),
        failure_candidate=degraded,
    )


def add_day09_vae(rows) -> None:
    if not DAY09_VAE.exists():
        return

    df = pd.read_csv(DAY09_VAE)

    if (
        df.empty
        or "mean_error" not in df.columns
    ):
        return

    item = df.sort_values(
        "mean_error",
        ascending=False,
    ).iloc[0]

    action = safe_text(
        item.get("action_label", "UNKNOWN")
    )

    add_case(
        rows,
        case_id=f"D09_VAE_{action}",
        source_day="DAY09",
        category="VAE_REFERENCE_DIFFERENCE",
        severity="OBSERVE",
        symptom=(
            "wave Reference와 가장 큰 "
            "평균 차이를 보인 Action"
        ),
        evidence=(
            f"action={action}; "
            f"mean_error="
            f"{safe_text(item.get('mean_error', ''))}; "
            f"candidate_ratio="
            f"{safe_text(item.get('anomaly_candidate_ratio', ''))}"
        ),
        failure_candidate=False,
    )


def add_fallback_case(rows) -> None:
    if any(
        int(row["failure_candidate"]) == 1
        for row in rows
    ):
        return

    if not DAY06_PREDICTIONS.exists():
        return

    df = pd.read_csv(DAY06_PREDICTIONS)

    if (
        df.empty
        or "confidence" not in df.columns
    ):
        return

    item = df.sort_values(
        "confidence",
        ascending=True,
    ).iloc[0]

    clip_id = safe_text(
        item.get("clip_id", "UNKNOWN")
    )

    add_case(
        rows,
        case_id=f"D06_LOWEST_CONF_{clip_id}",
        source_day="DAY06",
        category="ACTION",
        severity="PRACTICE",
        symptom=(
            "명확한 실패 후보가 없어 "
            "가장 낮은 Confidence 사례를 "
            "진단 연습용으로 선택"
        ),
        evidence=(
            f"true="
            f"{safe_text(item.get('true_label', ''))}; "
            f"pred="
            f"{safe_text(item.get('pred_label', ''))}; "
            f"confidence="
            f"{safe_text(item.get('confidence', ''))}"
        ),
        failure_candidate=True,
    )


def main() -> None:
    rows = []

    confidence_threshold = (
        load_runtime_confidence_threshold()
    )

    add_day05(rows)
    add_day06(
        rows,
        confidence_threshold,
    )
    add_day07(rows)
    add_day08(rows)
    add_day09_augmentation(rows)
    add_day09_vae(rows)
    add_fallback_case(rows)

    if not rows:
        raise RuntimeError(
            "Failure Catalog에 사용할 "
            "이전 결과가 없습니다."
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        [
            "failure_candidate",
            "source_day",
            "case_id",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        result[
            [
                "case_id",
                "source_day",
                "category",
                "severity",
                "failure_candidate",
                "symptom",
                "evidence",
            ]
        ].to_string(
            index=False
        )
    )
    print()
    print(
        "Failure candidates:",
        int(
            result[
                "failure_candidate"
            ].sum()
        ),
    )
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()