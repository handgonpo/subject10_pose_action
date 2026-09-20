
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

DEFAULT_OUTPUT_PATH = (
    ROOT
    / "data"
    / "day05"
    / "manifests"
    / "approved_clip_manifest.csv"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--review-path",
        default=str(
            DEFAULT_REVIEW_PATH
        ),
    )

    parser.add_argument(
        "--output-path",
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    review_path = Path(
        args.review_path
    )

    output_path = Path(
        args.output_path
    )

    if not review_path.exists():
        raise FileNotFoundError(
            review_path
        )

    df = pd.read_csv(
        review_path
    )

    # 자동 QA 결과 정리
    qa_status = (
        df["qa_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Human QA 결과 정리
    decisions = (
        df["human_decision"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # CHECK인데 아직 Human QA를 하지 않은 Clip이 있는지 확인
    pending_check = (
        (qa_status == "CHECK")
        & (
            (decisions == "")
            | (decisions == "UNREVIEWED")
        )
    )

    if pending_check.any():
        pending = df.loc[
            pending_check,
            [
                "clip_id",
                "qa_status",
                "human_decision",
            ],
        ]

        print()
        print(
            "[ERROR] Human QA가 끝나지 않은 "
            "CHECK Clip이 있습니다."
        )
        print()

        print(
            pending.to_string(
                index=False
            )
        )

        raise SystemExit(1)


    # 실제 다음 단계에서 사용할 승인 데이터 선택
    approved_mask = (
        # 자동 QA에서 통과한 Clip
        (
            (qa_status == "AUTO_PASS")
            & (
                decisions.isin(
                    [
                        "",
                        "UNREVIEWED",
                        "PASS",
                    ]
                )
            )
        )
        |
        # 자동 QA에서는 CHECK였지만
        # 사람이 확인하여 PASS한 Clip
        (
            (qa_status == "CHECK")
            & (decisions == "PASS")
        )
    )

    approved = df[
        approved_mask
    ].copy()

    if approved.empty:
        raise RuntimeError(
            "PASS 데이터가 없습니다."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    approved.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Approved clips: "
        f"{len(approved)}"
    )

    print()
    print("[By Subject]")
    print(
        approved.groupby(
            "subject"
        ).size()
    )

    print()
    print("[By Action]")
    print(
        approved.groupby(
            "action_label"
        ).size()
    )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()