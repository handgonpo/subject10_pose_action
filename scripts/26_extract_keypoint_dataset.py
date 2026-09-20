
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (
    load_config,
    resolve_device,
)
from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
)


DEFAULT_VIDEO_ROOT = (
    ROOT / "data" / "day05" / "videos"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT / "data" / "day05" / "keypoints"
)

ACTION_FILE = (
    ROOT / "configs" / "action_classes.csv"
)

DEFAULT_MANIFEST_PATH = (
    ROOT
    / "data"
    / "day05"
    / "manifests"
    / "keypoint_dataset_manifest.csv"
)

FILENAME_PATTERN = re.compile(
    r"^(S\d{2})_"
    r"(A\d{3})_"
    r"(T\d{2})\.mp4$",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video-root",
        default=str(
            DEFAULT_VIDEO_ROOT
        ),
    )

    parser.add_argument(
        "--output-root",
        default=str(
            DEFAULT_OUTPUT_ROOT
        ),
    )

    parser.add_argument(
        "--manifest-path",
        default=str(
            DEFAULT_MANIFEST_PATH
        ),
    )

    parser.add_argument(
        "--expected-clips",
        type=int,
        default=18,
        help=(
            "기본 본수업은 18개. "
            "Challenge에서는 3을 사용합니다."
        ),
    )

    return parser.parse_args()


def load_action_map() -> dict[str, str]:
    with ACTION_FILE.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    action_map = {
        row["action_id"]
        .strip()
        .upper():
        row["action_label"]
        .strip()
        for row in rows
    }

    expected = {
        "A001",
        "A002",
        "A003",
        "A004",
        "A005",
        "A006",
    }

    if set(action_map) != expected:
        raise RuntimeError(
            "action_classes.csv의 "
            "A001~A006 정의를 "
            "확인하세요."
        )

    return action_map


def build_keypoint_columns() -> list[str]:
    columns = []

    for name in COCO_KEYPOINT_NAMES:
        columns.extend(
            [
                f"{name}_x_px",
                f"{name}_y_px",
                f"{name}_x_norm",
                f"{name}_y_norm",
                f"{name}_conf",
            ]
        )

    return columns


def parse_clip_name(
    path: Path,
    action_map: dict[str, str],
):
    match = FILENAME_PATTERN.match(
        path.name
    )

    if match is None:
        raise ValueError(
            f"파일명 규칙 오류: "
            f"{path.name}"
        )

    subject = match.group(1).upper()
    action_id = match.group(2).upper()
    take = match.group(3).upper()

    if path.parent.name != subject:
        raise ValueError(
            "Subject 폴더와 파일명이 "
            f"다릅니다: {path}"
        )

    action_label = action_map.get(
        action_id
    )

    if action_label is None:
        raise ValueError(
            f"등록되지 않은 Action: "
            f"{action_id}"
        )

    return (
        subject,
        action_id,
        action_label,
        take,
    )


def select_person(result):
    if (
        result.boxes is None
        or result.keypoints is None
        or result.keypoints.xy is None
        or len(result.boxes) == 0
        or len(result.keypoints.xy) == 0
    ):
        return None

    box_conf = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    person_count = min(
        len(box_conf),
        len(result.keypoints.xy),
    )

    if person_count == 0:
        return None

    return int(
        np.argmax(
            box_conf[:person_count]
        )
    )


def process_clip(
    model,
    clip_path: Path,
    action_map: dict[str, str],
    config: dict,
    device,
    output_root: Path,
) -> dict:
    (
        subject,
        action_id,
        action_label,
        take,
    ) = parse_clip_name(
        clip_path,
        action_map,
    )

    capture = cv2.VideoCapture(
        str(clip_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: "
            f"{clip_path}"
        )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )
    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )
    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )
    metadata_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if (
        fps <= 0
        or width <= 0
        or height <= 0
    ):
        capture.release()
        raise RuntimeError(
            "영상 Metadata를 확인할 수 "
            f"없습니다: {clip_path}"
        )

    output_dir = (
        output_root
        / subject
        / action_label
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{clip_path.stem}.csv"
    )

    keypoint_columns = (
        build_keypoint_columns()
    )

    fieldnames = [
        "clip_id",
        "frame_id",
        "time_sec",
        "subject",
        "action_id",
        "action_label",
        "take",
        "person_count",
        "selected_box_conf",
        "pose_detected",
        *keypoint_columns,
    ]

    rows = []
    frame_id = 0
    detected_frames = 0
    single_person_frames = 0
    frame_mean_confidences = []

    while True:
        success, frame = capture.read()

        if not success:
            break

        result = model.predict(
            source=frame,
            imgsz=config["image_size"],
            conf=config[
                "person_confidence"
            ],
            device=device,
            verbose=False,
        )[0]

        person_count = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        if person_count == 1:
            single_person_frames += 1

        row = {
            "clip_id": clip_path.stem,
            "frame_id": frame_id,
            "time_sec": round(
                frame_id / fps,
                4,
            ),
            "subject": subject,
            "action_id": action_id,
            "action_label": action_label,
            "take": take,
            "person_count": person_count,
            "selected_box_conf": "",
            "pose_detected": 0,
        }

        for column in keypoint_columns:
            row[column] = ""

        selected_index = select_person(
            result
        )

        if selected_index is not None:
            xy = (
                result.keypoints.xy[
                    selected_index
                ]
                .detach()
                .cpu()
                .numpy()
            )

            if len(xy) != len(
                COCO_KEYPOINT_NAMES
            ):
                capture.release()
                raise RuntimeError(
                    "COCO 17 Keypoint 수가 "
                    f"아닙니다: "
                    f"{clip_path.name}"
                )

            if (
                result.keypoints.conf
                is not None
            ):
                kp_conf = (
                    result.keypoints.conf[
                        selected_index
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                kp_conf = np.ones(
                    len(
                        COCO_KEYPOINT_NAMES
                    ),
                    dtype=np.float32,
                )

            box_conf = float(
                result.boxes.conf[
                    selected_index
                ]
                .detach()
                .cpu()
                .item()
            )

            row[
                "selected_box_conf"
            ] = round(
                box_conf,
                4,
            )
            row["pose_detected"] = 1

            detected_frames += 1

            frame_mean_confidences.append(
                float(
                    np.mean(kp_conf)
                )
            )

            for index, name in enumerate(
                COCO_KEYPOINT_NAMES
            ):
                x_px, y_px = xy[index]

                x_norm = (
                    float(x_px) / width
                )
                y_norm = (
                    float(y_px) / height
                )
                confidence = float(
                    kp_conf[index]
                )

                row[
                    f"{name}_x_px"
                ] = round(
                    float(x_px),
                    3,
                )
                row[
                    f"{name}_y_px"
                ] = round(
                    float(y_px),
                    3,
                )
                row[
                    f"{name}_x_norm"
                ] = round(
                    x_norm,
                    6,
                )
                row[
                    f"{name}_y_norm"
                ] = round(
                    y_norm,
                    6,
                )
                row[
                    f"{name}_conf"
                ] = round(
                    confidence,
                    4,
                )

        rows.append(row)
        frame_id += 1

        if frame_id % 60 == 0:
            print(
                f"  frame {frame_id}"
            )

    capture.release()

    if frame_id == 0:
        raise RuntimeError(
            "읽은 Frame이 없습니다: "
            f"{clip_path}"
        )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    pose_ratio = (
        detected_frames / frame_id
    )
    single_person_ratio = (
        single_person_frames / frame_id
    )

    mean_kp_conf = (
        float(
            np.mean(
                frame_mean_confidences
            )
        )
        if frame_mean_confidences
        else 0.0
    )

    return {
        "clip_id": clip_path.stem,
        "video_path": str(
            clip_path.relative_to(ROOT)
        ),
        "keypoint_csv": str(
            output_path.relative_to(ROOT)
        ),
        "subject": subject,
        "action_id": action_id,
        "action_label": action_label,
        "take": take,
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "metadata_frames": (
            metadata_frames
        ),
        "read_frames": frame_id,
        "pose_detected_frames": (
            detected_frames
        ),
        "pose_ratio": round(
            pose_ratio,
            4,
        ),
        "single_person_ratio": round(
            single_person_ratio,
            4,
        ),
        "mean_keypoint_conf": round(
            mean_kp_conf,
            4,
        ),
    }


def main() -> None:
    args = parse_args()

    video_root = Path(
        args.video_root
    )
    output_root = Path(
        args.output_root
    )
    manifest_path = Path(
        args.manifest_path
    )

    if args.expected_clips < 1:
        raise ValueError(
            "--expected-clips는 "
            "1 이상이어야 합니다."
        )

    if not ACTION_FILE.exists():
        raise FileNotFoundError(
            ACTION_FILE
        )

    if not video_root.exists():
        raise FileNotFoundError(
            video_root
        )

    action_map = load_action_map()
    config = load_config()
    device = resolve_device(
        config["device"]
    )

    clips = sorted(
        video_root.rglob("*.mp4")
    )

    if len(clips) != args.expected_clips:
        raise RuntimeError(
            "입력 MP4 개수가 예상과 다릅니다. "
            f"expected={args.expected_clips}, "
            f"actual={len(clips)}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = YOLO(
        config["pose_model"]
    )

    manifest_rows = []

    print(f"Clips: {len(clips)}")
    print()

    for index, clip in enumerate(
        clips,
        start=1,
    ):
        print(
            f"[{index}/{len(clips)}] "
            f"{clip.name}"
        )

        item = process_clip(
            model=model,
            clip_path=clip,
            action_map=action_map,
            config=config,
            device=device,
            output_root=output_root,
        )

        manifest_rows.append(item)

        print(
            "  pose_ratio="
            f"{item['pose_ratio']} "
            "mean_conf="
            f"{item['mean_keypoint_conf']}"
        )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                manifest_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            manifest_rows
        )

    print()
    print(
        "Keypoint extraction: COMPLETE"
    )
    print(
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()