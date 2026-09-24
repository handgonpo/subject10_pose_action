from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR = (
    ROOT
    / "reports"
    / "day07"
    / "ip_webcam"
)

DEFAULT_OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day07"
    / "ip_webcam"
    / "ip_webcam_runtime_summary.csv"
)

REQUIRED_COLUMNS = {
    "expected_label",
    "prediction_updated",
    "threshold_label",
    "stable_label",
    "raw_label",
    "confidence",
    "processing_fps",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log",
        default=None,
        help=(
            "특정 Runtime Log를 직접 지정합니다. "
            "생략하면 입력 폴더에서 가장 최근의 "
            "분석 가능한 Log를 자동으로 찾습니다."
        ),
    )

    parser.add_argument(
        "--input-dir",
        default=str(
            DEFAULT_INPUT_DIR
        ),
        help="IP Webcam Runtime Log가 저장된 폴더",
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
        help="Action별 Runtime 요약 CSV 저장 경로",
    )

    return parser.parse_args()


def resolve_path(
    value: str | Path,
) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = ROOT / path

    return path


def read_runtime_log(
    path: Path,
):
    df = pd.read_csv(
        path
    )

    missing = sorted(
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "필수 Column이 없습니다: "
            + ", ".join(missing)
        )

    expected = (
        df[
            "expected_label"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    prediction_mask = (
        pd.to_numeric(
            df[
                "prediction_updated"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        == 1
    )

    target_mask = (
        expected
        != ""
    )

    usable = df[
        prediction_mask
        & target_mask
    ].copy()

    if not usable.empty:
        usable[
            "expected_label"
        ] = (
            usable[
                "expected_label"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    diagnostics = {
        "rows": int(
            len(df)
        ),
        "target_rows": int(
            target_mask.sum()
        ),
        "prediction_rows": int(
            prediction_mask.sum()
        ),
        "usable_rows": int(
            len(usable)
        ),
    }

    return (
        df,
        usable,
        diagnostics,
    )


def runtime_log_candidates(
    input_dir: Path,
    output_path: Path,
):
    files = list(
        input_dir.glob(
            "ip_webcam_w*_c*_s*_*.csv"
        )
    )

    if not files:
        files = list(
            input_dir.glob(
                "ip_webcam_*.csv"
            )
        )

    cleaned = []

    for path in files:
        if (
            path.resolve()
            == output_path.resolve()
        ):
            continue

        lower_name = (
            path.name.lower()
        )

        if (
            "summary" in lower_name
            or "comparison" in lower_name
        ):
            continue

        cleaned.append(
            path
        )

    return sorted(
        cleaned,
        key=lambda p: (
            p.stat().st_mtime,
            p.name,
        ),
        reverse=True,
    )


def print_log_diagnostic(
    path: Path,
    diagnostic: dict,
    prefix: str = "",
) -> None:
    print(
        f"{prefix}{path.name} | "
        f"rows={diagnostic['rows']} | "
        f"target_rows={diagnostic['target_rows']} | "
        f"prediction_rows={diagnostic['prediction_rows']} | "
        f"usable_rows={diagnostic['usable_rows']}"
    )


def choose_log(
    value,
    input_dir: Path,
    output_path: Path,
):
    if value:
        path = resolve_path(
            value
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        try:
            (
                df,
                usable,
                diagnostic,
            ) = read_runtime_log(
                path
            )
        except Exception as exc:
            print(
                f"[ERROR] Log를 읽을 수 없습니다: "
                f"{path}"
            )
            print(
                f"원인: {exc}"
            )
            raise SystemExit(1)

        if usable.empty:
            print()
            print(
                "[STOP] 지정한 Log에는 "
                "Target이 표시된 Prediction 결과가 없습니다."
            )
            print_log_diagnostic(
                path,
                diagnostic,
                prefix="  ",
            )
            print()
            print(
                "44번 실시간 실행에서 다음을 확인하세요."
            )
            print(
                "1) 카메라 창을 클릭하여 키 입력이 들어가게 합니다."
            )
            print(
                "2) 1~6 키를 눌러 Target을 선택합니다."
            )
            print(
                "3) 화면의 Target이 '-'가 아니라 Action 이름인지 확인합니다."
            )
            print(
                "4) WARMING_UP이 끝날 때까지 동작을 충분히 수행합니다."
            )
            print(
                "5) Prediction이 나온 뒤 Q로 종료합니다."
            )
            raise SystemExit(1)

        return (
            path,
            df,
            usable,
        )

    files = runtime_log_candidates(
        input_dir,
        output_path,
    )

    if not files:
        print(
            "[STOP] IP Webcam Runtime Log가 없습니다."
        )
        print(
            f"확인 폴더: {input_dir}"
        )
        raise SystemExit(1)

    skipped = []

    for path in files:
        try:
            (
                df,
                usable,
                diagnostic,
            ) = read_runtime_log(
                path
            )
        except Exception as exc:
            skipped.append(
                (
                    path,
                    {
                        "rows": 0,
                        "target_rows": 0,
                        "prediction_rows": 0,
                        "usable_rows": 0,
                    },
                    str(exc),
                )
            )
            continue

        if not usable.empty:
            if skipped:
                print(
                    "[INFO] 가장 최근 Log 중 일부는 "
                    "분석 조건을 만족하지 않아 건너뛰었습니다."
                )

                for (
                    skipped_path,
                    skipped_diag,
                    reason,
                ) in skipped:
                    print_log_diagnostic(
                        skipped_path,
                        skipped_diag,
                        prefix="  SKIP: ",
                    )

                    if reason:
                        print(
                            f"        reason={reason}"
                        )

                print()

            print(
                "[SELECTED] 가장 최근의 분석 가능한 Log"
            )
            print(
                f"  {path}"
            )

            return (
                path,
                df,
                usable,
            )

        skipped.append(
            (
                path,
                diagnostic,
                "",
            )
        )

    print()
    print(
        "[STOP] 분석 가능한 IP Webcam Runtime Log가 없습니다."
    )
    print(
        "현재 Log들을 확인했지만 "
        "'Target이 표시된 실제 Prediction'이 한 번도 없습니다."
    )
    print()

    for (
        skipped_path,
        skipped_diag,
        reason,
    ) in skipped:
        print_log_diagnostic(
            skipped_path,
            skipped_diag,
            prefix="  ",
        )

        if reason:
            print(
                f"      reason={reason}"
            )

    print()
    print(
        "다시 44번 실시간 실행을 진행하세요."
    )
    print(
        "1) 카메라 창을 클릭합니다."
    )
    print(
        "2) 1~6 키 중 하나를 눌러 Target을 선택합니다."
    )
    print(
        "3) 화면에서 Target이 Action 이름으로 바뀌었는지 확인합니다."
    )
    print(
        "4) WARMING_UP 이후 Prediction이 나오도록 "
        "동작을 충분히 수행합니다."
    )
    print(
        "5) Q로 종료하여 새 CSV를 저장합니다."
    )

    raise SystemExit(1)


def label_changes(
    series: pd.Series,
) -> int:
    clean = (
        series
        .fillna("")
        .astype(str)
    )

    if clean.empty:
        return 0

    return int(
        (
            clean
            != clean.shift(1)
        ).sum()
        - 1
    )


def main() -> None:
    args = parse_args()

    input_dir = resolve_path(
        args.input_dir
    )

    output_path = resolve_path(
        args.output
    )

    (
        log_path,
        _,
        usable,
    ) = choose_log(
        args.log,
        input_dir,
        output_path,
    )

    if "segment_id" not in usable.columns:
        usable[
            "segment_id"
        ] = 0

    rows = []

    for label, group in (
        usable.groupby(
            "expected_label"
        )
    ):
        segment_groups = list(
            group.groupby(
                "segment_id"
            )
        )

        raw_changes = sum(
            label_changes(
                segment[
                    "raw_label"
                ]
            )
            for _, segment
            in segment_groups
        )

        stable_changes = sum(
            label_changes(
                segment[
                    "stable_label"
                ]
            )
            for _, segment
            in segment_groups
        )

        fps_numeric = pd.to_numeric(
            group[
                "processing_fps"
            ],
            errors="coerce",
        )

        fps_values = fps_numeric[
            fps_numeric
            > 0
        ]

        confidence_numeric = (
            pd.to_numeric(
                group[
                    "confidence"
                ],
                errors="coerce",
            )
        )

        rows.append(
            {
                "expected_label": (
                    label
                ),
                "segments": (
                    len(
                        segment_groups
                    )
                ),
                "prediction_updates": (
                    len(group)
                ),
                "unknown_ratio": round(
                    float(
                        (
                            group[
                                "threshold_label"
                            ]
                            == "UNKNOWN"
                        ).mean()
                    ),
                    4,
                ),
                "stable_match_ratio": round(
                    float(
                        (
                            group[
                                "stable_label"
                            ]
                            == label
                        ).mean()
                    ),
                    4,
                ),
                "raw_label_changes": (
                    raw_changes
                ),
                "stable_label_changes": (
                    stable_changes
                ),
                "mean_confidence": round(
                    float(
                        confidence_numeric.mean()
                    ),
                    4,
                ),
                "mean_processing_fps": round(
                    float(
                        fps_values.mean()
                    ),
                    3,
                )
                if not fps_values.empty
                else 0.0,
                "window_seconds": (
                    group[
                        "window_seconds"
                    ].iloc[0]
                    if "window_seconds"
                    in group.columns
                    else ""
                ),
                "confidence_threshold": (
                    group[
                        "confidence_threshold"
                    ].iloc[0]
                    if "confidence_threshold"
                    in group.columns
                    else ""
                ),
                "stabilization_buffer": (
                    group[
                        "stabilization_buffer"
                    ].iloc[0]
                    if "stabilization_buffer"
                    in group.columns
                    else ""
                ),
                "stabilization_min_votes": (
                    group[
                        "stabilization_min_votes"
                    ].iloc[0]
                    if "stabilization_min_votes"
                    in group.columns
                    else ""
                ),
            }
        )

    result = pd.DataFrame(
        rows
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
    print(
        f"Log: {log_path}"
    )
    print()
    print(
        result.to_string(
            index=False
        )
    )
    print()
    print(
        "주의: stable_match_ratio는 "
        "수동 Target 표시와 겹치는 "
        "Sliding Window를 이용한 "
        "Runtime 확인값이며 "
        "정식 Test Accuracy가 아닙니다."
    )
    print(
        "같은 Action을 여러 번 수행한 경우 "
        "segment_id별 Label 변화량을 먼저 "
        "계산한 뒤 합산합니다."
    )
    print()
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()
