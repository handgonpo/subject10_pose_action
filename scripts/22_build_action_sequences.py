from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from src.common import (
    load_config,
    resolve_device,
)

from src.feature_utils import (
    normalize_by_shoulder_width,
    selected_joint_angles,
)

from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
)


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
    / "sequences"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default=None,
        help=(
            "특정 MP4만 처리할 때 파일명 입력. "
            "예: bend_return.mp4"
        ),
    )

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

    parser.add_argument(
        "--kp-conf",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help=(
            "0이면 끝까지 처리, "
            "양수이면 최대 처리 Row 수"
        ),
    )

    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help=(
            "1이면 모든 Frame, "
            "2이면 2 Frame마다 처리"
        ),
    )

    return parser.parse_args()


def make_feature_names() -> list[str]:
    names = []

    for keypoint_name in (
        COCO_KEYPOINT_NAMES
    ):
        names.append(
            f"{keypoint_name}_x"
        )

        names.append(
            f"{keypoint_name}_y"
        )

    names.extend(
        [
            "left_elbow_angle",
            "right_elbow_angle",
            "left_knee_angle",
            "right_knee_angle",
        ]
    )

    return names


def load_actions(
    action_file: Path,
) -> list[dict]:
    with action_file.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def make_empty_feature(
    feature_names: list[str],
) -> dict:
    return {
        name: ""
        for name in feature_names
    }


def process_clip(
    model,
    action: dict,
    config: dict,
    device,
    kp_threshold: float,
    max_frames: int,
    frame_step: int,
    video_dir: Path,
    output_dir: Path,
) -> None:
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

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    if fps <= 0:
        capture.release()

        raise RuntimeError(
            f"FPS를 확인할 수 없습니다: "
            f"{video_path}"
        )

    feature_names = (
        make_feature_names()
    )

    rows = []

    frame_id = -1
    processed_count = 0

    while True:
        success, frame = (
            capture.read()
        )

        if not success:
            break

        frame_id += 1

        if frame_id % frame_step != 0:
            continue

        processed_count += 1

        time_sec = (
            frame_id / fps
        )

        base_row = {
            "action_id": (
                action["action_id"]
            ),
            "action_label": (
                action["action_label"]
            ),
            "source": (
                action["video_file"]
            ),
            "frame_id": frame_id,
            "time_sec": round(
                time_sec,
                4,
            ),
            "pose_ok": 0,
            "feature_ok": 0,
            "mean_keypoint_conf": "",
            "low_keypoint_count": "",
            "hip_center_x": "",
            "hip_center_y": "",
            "shoulder_width": "",
        }

        result = model.predict(
            source=frame,
            imgsz=config[
                "image_size"
            ],
            conf=config[
                "person_confidence"
            ],
            device=device,
            verbose=False,
        )[0]

        if (
            result.boxes is None
            or result.keypoints
            is None
            or result.keypoints.xy
            is None
            or len(result.boxes) == 0
            or len(
                result.keypoints.xy
            ) == 0
        ):
            rows.append(
                {
                    **base_row,
                    **make_empty_feature(
                        feature_names
                    ),
                }
            )
        else:
            boxes_conf = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            xy = (
                result.keypoints.xy
                .cpu()
                .numpy()
            )

            person_count = min(
                len(boxes_conf),
                len(xy),
            )

            if person_count == 0:
                rows.append(
                    {
                        **base_row,
                        **make_empty_feature(
                            feature_names
                        ),
                    }
                )
            else:
                main_index = int(
                    np.argmax(
                        boxes_conf[
                            :person_count
                        ]
                    )
                )

                keypoints = xy[
                    main_index
                ]

                if (
                    result.keypoints.conf
                    is not None
                ):
                    confidence = (
                        result.keypoints.conf
                        .cpu()
                        .numpy()[
                            main_index
                        ]
                    )
                else:
                    confidence = (
                        np.ones(
                            17,
                            dtype=np.float32,
                        )
                    )

                mean_conf = float(
                    np.mean(
                        confidence
                    )
                )

                low_count = int(
                    np.sum(
                        confidence
                        < kp_threshold
                    )
                )

                detected_row = {
                    **base_row,
                    "pose_ok": 1,
                    "mean_keypoint_conf":
                        round(
                            mean_conf,
                            4,
                        ),
                    "low_keypoint_count":
                        low_count,
                }

                try:
                    normalized, (
                        center
                    ), scale = (
                        normalize_by_shoulder_width(
                            keypoints
                        )
                    )

                    angles = (
                        selected_joint_angles(
                            keypoints
                        )
                    )

                    feature_values = []

                    for point in normalized:
                        feature_values.extend(
                            [
                                float(
                                    point[0]
                                ),
                                float(
                                    point[1]
                                ),
                            ]
                        )

                    feature_values.extend(
                        [
                            angles[
                                "left_elbow_angle"
                            ],
                            angles[
                                "right_elbow_angle"
                            ],
                            angles[
                                "left_knee_angle"
                            ],
                            angles[
                                "right_knee_angle"
                            ],
                        ]
                    )

                    if len(
                        feature_values
                    ) != 38:
                        raise ValueError(
                            "Frame Feature 수가 "
                            "38개가 아닙니다."
                        )

                    if not np.all(
                        np.isfinite(
                            feature_values
                        )
                    ):
                        raise ValueError(
                            "Frame Feature에 "
                            "NaN 또는 Inf가 있습니다."
                        )

                    feature_row = {
                        name: round(
                            float(value),
                            6,
                        )
                        for name, value
                        in zip(
                            feature_names,
                            feature_values,
                        )
                    }

                    rows.append(
                        {
                            **detected_row,
                            "feature_ok": 1,
                            "hip_center_x":
                                round(
                                    float(
                                        center[0]
                                    ),
                                    4,
                                ),
                            "hip_center_y":
                                round(
                                    float(
                                        center[1]
                                    ),
                                    4,
                                ),
                            "shoulder_width":
                                round(
                                    float(scale),
                                    4,
                                ),
                            **feature_row,
                        }
                    )

                except ValueError:
                    rows.append(
                        {
                            **detected_row,
                            **make_empty_feature(
                                feature_names
                            ),
                        }
                    )

        if (
            max_frames > 0
            and processed_count
            >= max_frames
        ):
            break

        if (
            processed_count % 60
            == 0
        ):
            print(
                f"  processed: "
                f"{processed_count}"
            )

    capture.release()

    if not rows:
        raise RuntimeError(
            f"처리된 Frame이 없습니다: "
            f"{video_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / (
            f"{action['action_label']}"
            "_sequence.csv"
        )
    )

    fieldnames = [
        "action_id",
        "action_label",
        "source",
        "frame_id",
        "time_sec",
        "pose_ok",
        "feature_ok",
        "mean_keypoint_conf",
        "low_keypoint_count",
        "hip_center_x",
        "hip_center_y",
        "shoulder_width",
        *feature_names,
    ]

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

    pose_ok_count = sum(
        int(row["pose_ok"])
        for row in rows
    )

    feature_ok_count = sum(
        int(row["feature_ok"])
        for row in rows
    )

    print(
        f"[SAVED] "
        f"{action['action_label']} "
        f"rows={len(rows)} "
        f"pose_ok={pose_ok_count} "
        f"feature_ok="
        f"{feature_ok_count}"
    )

    print(
        f"        {output_path}"
    )


def main() -> None:
    args = parse_args()

    if args.frame_step < 1:
        raise ValueError(
            "--frame-step은 "
            "1 이상이어야 합니다."
        )

    if args.max_frames < 0:
        raise ValueError(
            "--max-frames는 "
            "0 이상이어야 합니다."
        )

    if not (
        0.0
        <= args.kp_conf
        <= 1.0
    ):
        raise ValueError(
            "--kp-conf는 "
            "0~1 범위여야 합니다."
        )

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

    actions = load_actions(
        action_file
    )

    if not actions:
        raise RuntimeError(
            "Action 정의가 없습니다."
        )

    if args.source is not None:
        actions = [
            action
            for action in actions
            if action["video_file"]
            == args.source
        ]

        if not actions:
            raise RuntimeError(
                "설정 파일에서 "
                f"{args.source}를 "
                "찾지 못했습니다."
            )

    config = load_config()

    device = resolve_device(
        config["device"]
    )

    model = YOLO(
        config["pose_model"]
    )

    for action in actions:
        print()
        print(
            f"Processing: "
            f"{action['video_file']}"
        )

        process_clip(
            model=model,
            action=action,
            config=config,
            device=device,
            kp_threshold=args.kp_conf,
            max_frames=args.max_frames,
            frame_step=args.frame_step,
            video_dir=video_dir,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()