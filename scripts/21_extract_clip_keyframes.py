from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ACTION_FILE = (
    ROOT
    / "configs"
    / "action_classes.csv"
)

DEFAULT_VIDEO_DIR = (
    ROOT
    / "data"
    / "day04"
    / "videos"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "reports"
    / "day04"
    / "keyframes"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--action-file",
        default=str(
            DEFAULT_ACTION_FILE
        ),
    )

    parser.add_argument(
        "--video-dir",
        default=str(
            DEFAULT_VIDEO_DIR
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    return parser.parse_args()


def read_actions(
    action_file: Path,
) -> list[dict]:
    with action_file.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def read_frame_at(
    video_path: Path,
    frame_index: int,
):
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: "
            f"{video_path}"
        )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_index,
    )

    success, frame = (
        capture.read()
    )

    capture.release()

    if (
        success
        and frame is not None
    ):
        return frame

    # 일부 Codec에서는 특정 Frame으로
    # 바로 이동하는 기능이 불안정할 수 있습니다.
    # 이 경우 처음부터 순서대로 다시 읽습니다.
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"영상을 다시 열 수 없습니다: "
            f"{video_path}"
        )

    frame = None

    for _ in range(
        frame_index + 1
    ):
        success, frame = (
            capture.read()
        )

        if not success:
            capture.release()

            raise RuntimeError(
                "Frame을 읽지 못했습니다: "
                f"{video_path.name} "
                f"frame={frame_index}"
            )

    capture.release()

    return frame


def main() -> None:
    args = parse_args()

    action_file = Path(
        args.action_file
    )

    video_dir = Path(
        args.video_dir
    )

    output_dir = Path(
        args.output_dir
    )

    if not action_file.exists():
        raise FileNotFoundError(
            action_file
        )

    if not video_dir.exists():
        raise FileNotFoundError(
            video_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    actions = read_actions(
        action_file
    )

    if not actions:
        raise RuntimeError(
            "Action 정의가 없습니다."
        )

    for action in actions:
        video_path = (
            video_dir
            / action["video_file"]
        )

        if not video_path.exists():
            raise FileNotFoundError(
                video_path
            )

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"영상을 열 수 없습니다: "
                f"{video_path}"
            )

        total_frames = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        capture.release()

        if total_frames < 1:
            raise RuntimeError(
                "Frame 수를 확인할 수 "
                f"없습니다: {video_path}"
            )

        positions = {
            "start": 0,
            "middle": (
                total_frames // 2
            ),
            "end": (
                total_frames - 1
            ),
        }

        for name, frame_index in (
            positions.items()
        ):
            output_path = (
                output_dir
                / (
                    f"{action['action_label']}"
                    f"_{name}.jpg"
                )
            )

            frame = read_frame_at(
                video_path,
                frame_index,
            )

            saved = cv2.imwrite(
                str(output_path),
                frame,
            )

            if not saved:
                raise RuntimeError(
                    "Frame을 저장하지 "
                    "못했습니다: "
                    f"{output_path}"
                )

        print(
            f"[OK] "
            f"{action['action_label']} "
            f"frames={total_frames}"
        )

    print()
    print(
        f"Saved: {output_dir}"
    )


if __name__ == "__main__":
    main()