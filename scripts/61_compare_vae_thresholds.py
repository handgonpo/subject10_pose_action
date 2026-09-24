from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "day09_vae.json"
NORMAL_ERROR_PATH = (
    ROOT
    / "reports"
    / "day09"
    / "vae"
    / "validation_normal_errors.csv"
)
ALL_SCORE_PATH = (
    ROOT
    / "reports"
    / "day09"
    / "vae"
    / "validation_all_window_scores.csv"
)
OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day09"
    / "experiments"
    / "vae_threshold_comparison.csv"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--percentiles",
        default=None,
        help=(
            "예: 95,99. 생략하면 "
            "day09_vae.json의 thresholds_compare 사용"
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            OUTPUT_PATH
        ),
    )

    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    args = parse_args()

    config = load_json(CONFIG_PATH)
    normal = pd.read_csv(NORMAL_ERROR_PATH)
    all_scores = pd.read_csv(ALL_SCORE_PATH)

    normal_errors = normal[
        "reconstruction_error"
    ].to_numpy(dtype=float)

    if len(normal_errors) == 0:
        raise RuntimeError(
            "Validation Normal Error가 없습니다."
        )

    if args.percentiles:
        percentiles = [
            float(value.strip())
            for value in args.percentiles.split(",")
            if value.strip()
        ]
    else:
        percentiles = [
            float(value)
            for value in config[
                "thresholds_compare"
            ]
        ]

    if (
        not percentiles
        or any(
            not 0 <= value <= 100
            for value in percentiles
        )
    ):
        raise ValueError(
            "Percentile은 0~100 범위의 "
            "값이 1개 이상 필요합니다."
        )

    output_path = Path(
        args.output
    )

    if not output_path.is_absolute():
        output_path = ROOT / output_path

    rows = []

    for percentile in percentiles:
        threshold = float(
            np.percentile(
                normal_errors,
                percentile,
            )
        )

        scored = all_scores.copy()
        scored["candidate"] = (
            scored["reconstruction_error"] > threshold
        )

        grouped = (
            scored.groupby("action_label")["candidate"]
            .mean()
            .reset_index()
        )

        for _, item in grouped.iterrows():
            rows.append(
                {
                    "percentile": percentile,
                    "threshold": threshold,
                    "action_label": item[
                        "action_label"
                    ],
                    "candidate_ratio": float(
                        item["candidate"]
                    ),
                }
            )

    result = pd.DataFrame(rows)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(result.to_string(index=False))
    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()