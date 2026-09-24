from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LOG_DIR = (
    ROOT / "reports" / "day08" / "logs"
)

DEFAULT_OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day08"
    / "experiments"
    / "tracking_scenario_summary.csv"
)

DEFAULT_SCENARIOS = [
    "01_sbu_shaking_hands",
    "02_sbu_hugging",
    "03_sbu_kicking_challenge",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log-dir",
        default=str(
            DEFAULT_LOG_DIR
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
    )

    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=DEFAULT_SCENARIOS,
        help=(
            "비교할 Scenario stem. "
            "예: challenge_crossing "
            "challenge_exit_reentry"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_dir = Path(
        args.log_dir
    )

    output_path = Path(
        args.output
    )

    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir

    if not output_path.is_absolute():
        output_path = (
            ROOT / output_path
        )

    rows = []

    for scenario in args.scenarios:
        path = (
            log_dir
            / (
                f"{scenario}"
                "_tracking_summary.csv"
            )
        )

        if not path.exists():
            raise FileNotFoundError(
                "먼저 48번 분석을 "
                f"실행하세요: {path}"
            )

        df = pd.read_csv(
            path
        )

        if len(df) != 1:
            raise RuntimeError(
                "Tracking Summary가 "
                f"한 행이 아닙니다: {path}"
            )

        rows.append(
            df.iloc[0].to_dict()
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

    print(
        result.to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()