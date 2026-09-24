from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIR = (
    ROOT / "reports" / "day08" / "logs"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_path = Path(args.log)
    output_dir = Path(
        args.output_dir
    )

    if not log_path.is_absolute():
        log_path = ROOT / log_path

    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    if not log_path.exists():
        raise FileNotFoundError(log_path)

    df = pd.read_csv(log_path)

    if df.empty:
        raise RuntimeError("Tracking Log가 비어 있습니다.")

    valid = df[
        pd.to_numeric(
            df["track_id"],
            errors="coerce",
        ).fillna(-1) >= 0
    ].copy()

    total_frames = int(df["frame_id"].max()) + 1

    if valid.empty:
        tracked_frame_ratio = 0.0
        two_id_frame_ratio = 0.0
        max_ids_in_frame = 0
        unique_ids = []
        id_summary = pd.DataFrame(
            columns=[
                "track_id",
                "first_frame",
                "last_frame",
                "detected_frames",
                "mean_box_conf",
                "mean_keypoint_conf",
            ]
        )
    else:
        ids_per_frame = (
            valid.groupby("frame_id")["track_id"]
            .nunique()
        )

        tracked_frame_ratio = (
            ids_per_frame.index.nunique()
            / total_frames
        )

        two_id_frame_ratio = (
            (ids_per_frame == 2).sum()
            / total_frames
        )

        max_ids_in_frame = int(ids_per_frame.max())

        unique_ids = sorted(
            valid["track_id"]
            .astype(int)
            .unique()
            .tolist()
        )

        id_summary = (
            valid.groupby("track_id")
            .agg(
                first_frame=("frame_id", "min"),
                last_frame=("frame_id", "max"),
                detected_frames=("frame_id", "nunique"),
                mean_box_conf=("box_conf", "mean"),
                mean_keypoint_conf=(
                    "mean_keypoint_conf",
                    "mean",
                ),
            )
            .reset_index()
        )

    scenario = log_path.stem.replace(
        "_tracking",
        "",
    )

    summary = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "total_frames": total_frames,
                "unique_track_ids": len(unique_ids),
                "track_ids": "|".join(
                    str(value) for value in unique_ids
                ),
                "tracked_frame_ratio": round(
                    tracked_frame_ratio,
                    4,
                ),
                "two_id_frame_ratio": round(
                    two_id_frame_ratio,
                    4,
                ),
                "max_ids_in_frame": max_ids_in_frame,
                "extra_id_count": max(
                    0,
                    len(unique_ids) - 2,
                ),
            }
        ]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir
        / f"{scenario}_tracking_summary.csv"
    )

    id_path = (
        output_dir
        / f"{scenario}_track_id_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    id_summary.to_csv(
        id_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(summary.to_string(index=False))
    print()

    if not id_summary.empty:
        print(id_summary.to_string(index=False))
        print()

    print(
        "주의: Unique ID가 3개 이상이라고 해서 "
        "그 자체로 ID Switch가 확정되는 것은 아닙니다."
    )
    print(
        "Track Lost 후 새 ID가 생성된 경우와 "
        "실제 ID Switch를 결과영상에서 구분하세요."
    )
    print()
    print(f"Saved: {summary_path}")
    print(f"Saved: {id_path}")


if __name__ == "__main__":
    main()