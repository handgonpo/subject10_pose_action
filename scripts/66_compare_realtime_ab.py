from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day10"
    / "experiments"
    / "realtime_ab_comparison.csv"
)

REQUIRED_COLUMNS = {
    "prediction_updated",
    "threshold_label",
    "stable_label",
    "confidence",
}


# --------------------------------------------------
# 1. 실행 인자
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--before",
        required=True,
        help="Before Runtime CSV",
    )

    parser.add_argument(
        "--after",
        required=True,
        help="After Runtime CSV",
    )

    parser.add_argument(
        "--expected",
        default=None,
        help=(
            "Expected Action Label. "
            "생략하면 CSV의 expected_label을 사용합니다."
        ),
    )

    return parser.parse_args()


# --------------------------------------------------
# 2. 경로 처리
# --------------------------------------------------

def resolve_path(value: str) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def relative_text(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------
# 3. 같은 원본 영상인지 확인
# --------------------------------------------------

def source_key(path: Path) -> str:
    """
    예:
    S02_A006_T01_w3_c0.55_s5
    -> S02_A006_T01
    """

    return re.sub(
        r"_w[^_]+_c[^_]+_s\d+$",
        "",
        path.stem,
    )


# --------------------------------------------------
# 4. prediction_updated 값을 안전하게 bool로 변환
# --------------------------------------------------

def prediction_mask(
    series: pd.Series,
) -> pd.Series:

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {
            "1",
            "true",
            "yes",
        }
    )


# --------------------------------------------------
# 5. Expected Label 확인
# --------------------------------------------------

def infer_expected(
    df: pd.DataFrame,
    override,
) -> str:

    if override:
        return str(override)

    if "expected_label" not in df.columns:
        return ""

    values = (
        df["expected_label"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    values = values[
        values != ""
    ]

    if values.empty:
        return ""

    return str(
        values.iloc[0]
    )


# --------------------------------------------------
# 6. CSV 전체에서 설정값 읽기
# --------------------------------------------------

def first_valid_value(
    df: pd.DataFrame,
    column: str,
):
    """
    Prediction이 0회여도
    CSV 전체에서 Runtime 설정값을 읽습니다.
    """

    if column not in df.columns:
        return None

    values = (
        df[column]
        .dropna()
    )

    if values.empty:
        return None

    return values.iloc[0]


# --------------------------------------------------
# 7. 파일명에서도 Runtime 설정 읽기
# --------------------------------------------------

def parse_runtime_from_filename(
    path: Path,
) -> dict:
    """
    예:
    S02_A006_T01_w5_c0.55_s5.csv

    -> Window = 5
    -> Confidence = 0.55
    -> Stabilization = 5
    """

    match = re.search(
        r"_w(?P<window>[^_]+)"
        r"_c(?P<confidence>[^_]+)"
        r"_s(?P<stabilization>\d+)$",
        path.stem,
    )

    if not match:
        return {
            "window_seconds": None,
            "confidence_threshold": None,
            "stabilization_buffer": None,
        }

    try:
        window = float(
            match.group("window")
        )
    except ValueError:
        window = None

    try:
        confidence = float(
            match.group("confidence")
        )
    except ValueError:
        confidence = None

    try:
        stabilization = int(
            match.group("stabilization")
        )
    except ValueError:
        stabilization = None

    return {
        "window_seconds": window,
        "confidence_threshold": confidence,
        "stabilization_buffer": stabilization,
    }


# --------------------------------------------------
# 8. Runtime 설정값 확인
# --------------------------------------------------

def get_runtime_settings(
    df: pd.DataFrame,
    path: Path,
) -> dict:

    filename_settings = (
        parse_runtime_from_filename(path)
    )

    window = first_valid_value(
        df,
        "window_seconds",
    )

    confidence = first_valid_value(
        df,
        "confidence_threshold",
    )

    stabilization = first_valid_value(
        df,
        "stabilization_buffer",
    )

    # CSV에 값이 없으면 파일명에서 복구
    if window is None:
        window = filename_settings[
            "window_seconds"
        ]

    if confidence is None:
        confidence = filename_settings[
            "confidence_threshold"
        ]

    if stabilization is None:
        stabilization = filename_settings[
            "stabilization_buffer"
        ]

    # 숫자 타입 통일
    if window is not None:
        window = float(window)

    if confidence is not None:
        confidence = float(confidence)

    if stabilization is not None:
        stabilization = int(
            float(stabilization)
        )

    return {
        "window_seconds": window,
        "confidence_threshold": confidence,
        "stabilization_buffer": stabilization,
    }


# --------------------------------------------------
# 9. Stable Label 변화 횟수
# --------------------------------------------------

def label_changes(
    series: pd.Series,
) -> int:

    values = (
        series
        .fillna("")
        .astype(str)
        .tolist()
    )

    if len(values) < 2:
        return 0

    return sum(
        current != previous
        for previous, current
        in zip(
            values[:-1],
            values[1:],
        )
    )


# --------------------------------------------------
# 10. 첫 Prediction 시점
# --------------------------------------------------

def first_prediction_time(
    updates: pd.DataFrame,
):

    if updates.empty:
        return ""

    for column in [
        "video_time_sec",
        "elapsed_sec",
    ]:
        if column in updates.columns:

            value = pd.to_numeric(
                updates[column],
                errors="coerce",
            ).dropna()

            if not value.empty:
                return round(
                    float(value.iloc[0]),
                    4,
                )

    return ""


# --------------------------------------------------
# 11. 평균 FPS
# --------------------------------------------------

def mean_processing_fps(
    df: pd.DataFrame,
):

    if "processing_fps" not in df.columns:
        return ""

    values = pd.to_numeric(
        df["processing_fps"],
        errors="coerce",
    )

    values = values[
        values > 0
    ]

    if values.empty:
        return ""

    return round(
        float(values.mean()),
        3,
    )


# --------------------------------------------------
# 12. Runtime CSV 한 개 요약
# --------------------------------------------------

def summarize(
    run_name: str,
    path: Path,
    expected_override,
) -> dict:

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    missing = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{path.name}에 필수 열이 없습니다: "
            + ", ".join(
                sorted(missing)
            )
        )

    # Runtime 설정은 Prediction 여부와 관계없이
    # CSV 전체 또는 파일명에서 먼저 읽습니다.
    runtime_settings = (
        get_runtime_settings(
            df,
            path,
        )
    )

    expected = infer_expected(
        df,
        expected_override,
    )

    mask = prediction_mask(
        df["prediction_updated"]
    )

    updates = (
        df.loc[mask]
        .copy()
        .reset_index(drop=True)
    )

    fps = mean_processing_fps(
        df
    )

    # ----------------------------------------------
    # Prediction이 한 번도 없었던 경우
    # ----------------------------------------------

    if updates.empty:
        return {
            "run": run_name,
            "file": relative_text(path),
            "expected_label": expected,

            "prediction_updates": 0,
            "first_prediction_sec": "",

            # Prediction 자체가 없으므로
            # UNKNOWN 비율도 계산할 수 없음
            "unknown_ratio": "",

            "stable_label_changes": 0,
            "stable_match_ratio": "",

            "mean_confidence": "",
            "mean_processing_fps": fps,

            "window_seconds": (
                runtime_settings[
                    "window_seconds"
                ]
            ),
            "confidence_threshold": (
                runtime_settings[
                    "confidence_threshold"
                ]
            ),
            "stabilization_buffer": (
                runtime_settings[
                    "stabilization_buffer"
                ]
            ),
        }

    # ----------------------------------------------
    # UNKNOWN Ratio
    # ----------------------------------------------

    threshold_labels = (
        updates["threshold_label"]
        .fillna("")
        .astype(str)
    )

    unknown_ratio = float(
        (
            threshold_labels
            == "UNKNOWN"
        ).mean()
    )

    # ----------------------------------------------
    # Stable Label Changes
    # ----------------------------------------------

    stable_changes = label_changes(
        updates["stable_label"]
    )

    # ----------------------------------------------
    # Stable Match Ratio
    # ----------------------------------------------

    if expected:

        stable_labels = (
            updates["stable_label"]
            .fillna("")
            .astype(str)
        )

        stable_match_ratio = float(
            (
                stable_labels
                == expected
            ).mean()
        )

    else:
        stable_match_ratio = ""

    # ----------------------------------------------
    # Mean Confidence
    # ----------------------------------------------

    confidence_values = pd.to_numeric(
        updates["confidence"],
        errors="coerce",
    ).dropna()

    if confidence_values.empty:
        mean_confidence = ""
    else:
        mean_confidence = round(
            float(
                confidence_values.mean()
            ),
            4,
        )

    # ----------------------------------------------
    # 결과 반환
    # ----------------------------------------------

    return {
        "run": run_name,
        "file": relative_text(path),
        "expected_label": expected,

        "prediction_updates": int(
            len(updates)
        ),

        "first_prediction_sec": (
            first_prediction_time(
                updates
            )
        ),

        "unknown_ratio": round(
            unknown_ratio,
            4,
        ),

        "stable_label_changes": (
            stable_changes
        ),

        "stable_match_ratio": (
            round(
                stable_match_ratio,
                4,
            )
            if stable_match_ratio != ""
            else ""
        ),

        "mean_confidence": (
            mean_confidence
        ),

        "mean_processing_fps": fps,

        "window_seconds": (
            runtime_settings[
                "window_seconds"
            ]
        ),

        "confidence_threshold": (
            runtime_settings[
                "confidence_threshold"
            ]
        ),

        "stabilization_buffer": (
            runtime_settings[
                "stabilization_buffer"
            ]
        ),
    }


# --------------------------------------------------
# 13. 값 비교용 정규화
# --------------------------------------------------

def same_value(
    left,
    right,
) -> bool:

    if left is None or right is None:
        return left == right

    try:
        return abs(
            float(left)
            - float(right)
        ) < 1e-9

    except (
        TypeError,
        ValueError,
    ):
        return left == right


# --------------------------------------------------
# 14. Main
# --------------------------------------------------

def main() -> None:

    args = parse_args()

    before_path = resolve_path(
        args.before
    )

    after_path = resolve_path(
        args.after
    )

    # 같은 원본 영상인지 확인
    if (
        source_key(before_path)
        != source_key(after_path)
    ):
        raise RuntimeError(
            "Before와 After는 같은 입력 영상에서 "
            "생성된 Runtime CSV여야 합니다."
        )

    before_summary = summarize(
        "before",
        before_path,
        args.expected,
    )

    after_summary = summarize(
        "after",
        after_path,
        args.expected,
    )

    # Expected Label 확인
    if (
        before_summary["expected_label"]
        != after_summary["expected_label"]
    ):
        raise RuntimeError(
            "Before와 After의 "
            "Expected Label이 다릅니다."
        )

    controlled_fields = [
        "window_seconds",
        "confidence_threshold",
        "stabilization_buffer",
    ]

    # 설정값을 읽지 못했다면
    # 잘못된 비교를 하지 않고 중단
    for field in controlled_fields:

        if (
            before_summary[field] is None
            or after_summary[field] is None
        ):
            raise RuntimeError(
                f"{field} 값을 확인할 수 없습니다. "
                "Runtime CSV 또는 파일명을 확인하세요."
            )

    # 정확히 어떤 값이 달라졌는지 확인
    changed = [
        field
        for field in controlled_fields
        if not same_value(
            before_summary[field],
            after_summary[field],
        )
    ]

    if len(changed) != 1:
        raise RuntimeError(
            "One Variable A/B 실험에서는 "
            "Window / Confidence / Stabilization 중 "
            "정확히 한 값만 달라야 합니다. "
            f"현재 변경 항목: {changed}"
        )

    changed_variable = (
        changed[0]
    )

    before_summary[
        "changed_variable"
    ] = changed_variable

    after_summary[
        "changed_variable"
    ] = changed_variable

    result = pd.DataFrame(
        [
            before_summary,
            after_summary,
        ]
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
        result.to_string(
            index=False
        )
    )

    print()

    print(
        "Changed variable:",
        changed_variable,
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()