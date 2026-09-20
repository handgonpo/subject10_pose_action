from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REVIEW_PATH = (
    ROOT
    / "data"
    / "day05"
    / "qa"
    / "keypoint_clip_review.csv"
)

ALLOWED = {
    "PASS",
    "RESHOOT",
    "EXCLUDE",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--review-path",
        default=str(
            DEFAULT_REVIEW_PATH
        ),
    )

    parser.add_argument(
        "--expected-clips",
        type=int,
        default=18,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    review_path = Path(
        args.review_path
    )

    if not review_path.exists():
        print(
            "[ERROR] Human QA 파일을 찾을 수 없습니다."
        )
        print(review_path)
        raise SystemExit(1)

    df = pd.read_csv(
        review_path
    )

    if len(df) != args.expected_clips:
        print(
            "[ERROR] Human QA Clip 수가 "
            "예상과 다릅니다."
        )
        print(
            f"expected={args.expected_clips}, "
            f"actual={len(df)}"
        )
        raise SystemExit(1)

    decisions = (
        df["human_decision"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # 아직 사람이 검토하지 않은 Clip
    unreviewed_mask = (
        (decisions == "")
        | (decisions == "UNREVIEWED")
    )

    if unreviewed_mask.any():
        unreviewed = df.loc[
            unreviewed_mask,
            [
                "clip_id",
                "qa_status",
                "human_decision",
            ],
        ]

        print()
        print(
            "[HUMAN QA NOT COMPLETE]"
        )
        print()
        print(
            unreviewed.to_string(
                index=False
            )
        )

        print()
        print(
            f"남은 Clip: "
            f"{int(unreviewed_mask.sum())}"
        )

        print()
        print(
            "원본 MP4를 확인한 뒤 "
            "human_decision을"
        )
        print(
            "PASS / RESHOOT / EXCLUDE 중 "
            "하나로 입력하세요."
        )

        raise SystemExit(1)

    # 허용되지 않은 값 검사
    invalid_mask = (
        ~decisions.isin(ALLOWED)
    )

    if invalid_mask.any():
        invalid = df.loc[
            invalid_mask,
            [
                "clip_id",
                "qa_status",
                "human_decision",
            ],
        ]

        print()
        print(
            "[INVALID HUMAN DECISION]"
        )
        print()
        print(
            invalid.to_string(
                index=False
            )
        )

        print()
        print(
            "human_decision은 "
            "PASS / RESHOOT / EXCLUDE만 "
            "사용할 수 있습니다."
        )

        raise SystemExit(1)

    print()
    print("[Human QA Result]")
    print(
        decisions.value_counts()
    )

    print()
    print(
        f"Reviewed clips: {len(df)}"
    )
    print(
        "Human review: OK"
    )


if __name__ == "__main__":
    main()