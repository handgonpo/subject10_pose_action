from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help="읽을 MP4 파일 경로",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = Path(args.source)

    if not video_path.exists():
        raise FileNotFoundError(
            f"파일이 없습니다: {video_path}"
        )

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: {video_path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    frame_number = 0

    while True:
        success, _frame = capture.read()

        if not success:
            break

        frame_number += 1

    capture.release()

    print(f"Video       : {video_path}")
    print(f"FPS         : {fps:.2f}")
    print(f"Frames      : {total_frames}")
    print(f"Read Frames : {frame_number}")
    print(f"Resolution  : {width} x {height}")

    if fps > 0:
        duration = total_frames / fps
        print(f"Duration    : {duration:.2f} sec")
    else:
        print("Duration    : FPS 정보를 읽지 못해 계산하지 않음")


if __name__ == "__main__":
    main()