from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (
    load_config,
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help="Pose를 적용할 MP4 파일",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="처리할 최대 Frame 수. 0이면 영상 끝까지 처리",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = Path(args.source)

    if not source.exists():
        raise FileNotFoundError(
            f"파일이 없습니다: {source}"
        )

    config = load_config()
    device = resolve_device(config["device"])

    model = YOLO(config["pose_model"])

    capture = cv2.VideoCapture(str(source))

    if not capture.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: {source}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    output_fps = fps if fps > 0 else 30.0

    success, frame = capture.read()

    if not success:
        capture.release()
        raise RuntimeError(
            "영상의 첫 Frame을 읽지 못했습니다."
        )

    height, width = frame.shape[:2]

    output_path = (
        ROOT
        / "reports"
        / "day01"
        / "pose_demo_output.mp4"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        output_fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            "결과 MP4 파일을 생성할 수 없습니다."
        )

    frame_number = 0

    while success:
        frame_number += 1

        results = model.predict(
            source=frame,
            imgsz=config["image_size"],
            conf=config["person_confidence"],
            device=device,
            verbose=False,
        )

        plotted = results[0].plot()

        writer.write(plotted)

        if (
            args.max_frames > 0
            and frame_number >= args.max_frames
        ):
            break

        success, frame = capture.read()

    capture.release()
    writer.release()

    print(f"Processed frames: {frame_number}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()