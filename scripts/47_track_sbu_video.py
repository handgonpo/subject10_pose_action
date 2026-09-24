from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


from src.common import load_config, resolve_device


DEFAULT_OUTPUT_VIDEO_DIR = (
    ROOT / "reports" / "day08" / "tracking"
)

DEFAULT_OUTPUT_LOG_DIR = (
    ROOT / "reports" / "day08" / "logs"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
    )

    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
    )

    parser.add_argument(
        "--output-video-dir",
        default=str(
            DEFAULT_OUTPUT_VIDEO_DIR
        ),
    )

    parser.add_argument(
        "--output-log-dir",
        default=str(
            DEFAULT_OUTPUT_LOG_DIR
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = Path(args.source)

    if not source.is_absolute():
        source = ROOT / source

    if not source.exists():
        raise FileNotFoundError(source)

    output_video_dir = Path(
        args.output_video_dir
    )
    output_log_dir = Path(
        args.output_log_dir
    )

    if not output_video_dir.is_absolute():
        output_video_dir = (
            ROOT / output_video_dir
        )

    if not output_log_dir.is_absolute():
        output_log_dir = (
            ROOT / output_log_dir
        )

    config = load_config()
    device = resolve_device(config["device"])

    model = YOLO(config["pose_model"])

    capture = cv2.VideoCapture(str(source))

    if not capture.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: {source}"
        )

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        capture.release()
        raise RuntimeError("영상 FPS를 확인할 수 없습니다.")

    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("영상 해상도를 확인할 수 없습니다.")

    output_video_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_video = (
        output_video_dir
        / f"{source.stem}_tracked.mp4"
    )

    output_log = (
        output_log_dir
        / f"{source.stem}_tracking.csv"
    )

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"결과 영상을 저장할 수 없습니다: {output_video}"
        )

    fieldnames = [
        "frame_id",
        "time_sec",
        "person_count",
        "track_id",
        "box_conf",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "mean_keypoint_conf",
    ]

    rows = []
    frame_id = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        result = model.track(
            source=frame,
            persist=True,
            tracker=args.tracker,
            imgsz=int(config["image_size"]),
            conf=float(config["person_confidence"]),
            device=device,
            verbose=False,
        )[0]

        annotated = result.plot()
        boxes = result.boxes
        keypoints = result.keypoints

        person_count = len(boxes) if boxes is not None else 0
        tracked_rows = 0

        if (
            boxes is not None
            and len(boxes) > 0
            and boxes.id is not None
        ):
            xyxy = boxes.xyxy.detach().cpu().numpy()
            box_conf = boxes.conf.detach().cpu().numpy()
            track_ids = (
                boxes.id.detach().cpu().numpy().astype(int)
            )

            kp_conf_all = None

            if (
                keypoints is not None
                and keypoints.conf is not None
            ):
                kp_conf_all = (
                    keypoints.conf.detach().cpu().numpy()
                )

            item_count = min(
                len(track_ids),
                len(xyxy),
                len(box_conf),
            )

            for index in range(item_count):
                x1, y1, x2, y2 = xyxy[index]
                track_id = int(track_ids[index])

                mean_kp_conf = ""

                if (
                    kp_conf_all is not None
                    and index < len(kp_conf_all)
                ):
                    mean_kp_conf = round(
                        float(np.mean(kp_conf_all[index])),
                        4,
                    )

                rows.append(
                    {
                        "frame_id": frame_id,
                        "time_sec": round(frame_id / fps, 4),
                        "person_count": person_count,
                        "track_id": track_id,
                        "box_conf": round(
                            float(box_conf[index]),
                            4,
                        ),
                        "bbox_x1": round(float(x1), 2),
                        "bbox_y1": round(float(y1), 2),
                        "bbox_x2": round(float(x2), 2),
                        "bbox_y2": round(float(y2), 2),
                        "mean_keypoint_conf": mean_kp_conf,
                    }
                )

                tracked_rows += 1

                cv2.putText(
                    annotated,
                    f"Track ID {track_id}",
                    (
                        int(x1),
                        max(25, int(y1) - 10),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

        if tracked_rows == 0:
            rows.append(
                {
                    "frame_id": frame_id,
                    "time_sec": round(frame_id / fps, 4),
                    "person_count": person_count,
                    "track_id": -1,
                    "box_conf": "",
                    "bbox_x1": "",
                    "bbox_y1": "",
                    "bbox_x2": "",
                    "bbox_y2": "",
                    "mean_keypoint_conf": "",
                }
            )

        writer.write(annotated)

        if frame_id % 10 == 0:
            current_ids = []

            if (
                boxes is not None
                and boxes.id is not None
            ):
                current_ids = (
                    boxes.id
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(int)
                    .tolist()
                )

            print(
                f"Frame {frame_id:3d} "
                f"IDs={current_ids}"
            )

        frame_id += 1

    capture.release()
    writer.release()

    if frame_id == 0:
        raise RuntimeError("처리한 Frame이 없습니다.")

    with output_log.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer_csv = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    print()
    print(f"Frames: {frame_id}")
    print(f"Video : {output_video}")
    print(f"Log   : {output_log}")


if __name__ == "__main__":
    main()