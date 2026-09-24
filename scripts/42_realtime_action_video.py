from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import cv2
import joblib
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

from src.realtime_utils import (
    FPSMeter,
    PredictionStabilizer,
    TimedFeatureBuffer,
    frame_feature_from_pose,
    predict_from_window,
    validate_runtime_values,
)


DAY07_CONFIG_PATH = (
    ROOT
    / "configs"
    / "day07_realtime.json"
)

TRAIN_INFO_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_training_info.json"
)

MODEL_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_action_baseline.joblib"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "reports"
    / "day07"
    / "saved_video"
)

ACTION_MAP = {
    "A001": "bend_return",
    "A002": "leg_raise_lower",
    "A003": "walk_turn_walk",
    "A004": "drink_return",
    "A005": "sit_stand",
    "A006": "wave",
}

FILENAME_PATTERN = re.compile(
    r"^S\d+_(A\d+)_T\d+\.mp4$",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--stabilization-buffer",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
        help=(
            "결과 MP4와 CSV를 "
            "저장할 폴더"
        ),
    )

    return parser.parse_args()


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def infer_expected_label(
    source: Path,
) -> str:
    match = FILENAME_PATTERN.match(
        source.name
    )

    if match is None:
        return ""

    action_id = (
        match.group(1)
        .upper()
    )

    return ACTION_MAP.get(
        action_id,
        "",
    )


def main() -> None:
    args = parse_args()

    source = Path(
        args.source
    )

    if not source.exists():
        raise FileNotFoundError(
            source
        )

    project_config = load_config()

    runtime_config = load_json(
        DAY07_CONFIG_PATH
    )

    training_info = load_json(
        TRAIN_INFO_PATH
    )

    sequence_steps = int(
        training_info[
            "sequence_steps"
        ]
    )

    window_seconds = (
        float(
            args.window_seconds
        )
        if args.window_seconds
        is not None
        else float(
            runtime_config[
                "window_seconds"
            ]
        )
    )

    confidence_threshold = (
        float(
            args.confidence
        )
        if args.confidence
        is not None
        else float(
            runtime_config[
                "confidence_threshold"
            ]
        )
    )

    stabilization_buffer = (
        int(
            args.stabilization_buffer
        )
        if args.stabilization_buffer
        is not None
        else int(
            runtime_config[
                "stabilization_buffer"
            ]
        )
    )

    if (
        args.stabilization_buffer
        is not None
    ):
        # Buffer 크기를 바꾸는 실험에서는
        # 과반수 기준도 함께 맞춥니다.
        stabilization_min_votes = (
            stabilization_buffer
            // 2
            + 1
        )
    else:
        stabilization_min_votes = int(
            runtime_config[
                "stabilization_min_votes"
            ]
        )

    output_dir = Path(
        args.output_dir
    )

    if not output_dir.is_absolute():
        output_dir = (
            ROOT
            / output_dir
        )

    validate_runtime_values(
        window_seconds=(
            window_seconds
        ),
        confidence_threshold=(
            confidence_threshold
        ),
        stabilization_buffer=(
            stabilization_buffer
        ),
        stabilization_min_votes=(
            stabilization_min_votes
        ),
    )

    device = resolve_device(
        project_config["device"]
    )

    action_model = joblib.load(
        MODEL_PATH
    )

    pose_model = YOLO(
        project_config[
            "pose_model"
        ]
    )

    capture = cv2.VideoCapture(
        str(source)
    )

    if not capture.isOpened():
        raise RuntimeError(
            "영상을 열 수 없습니다: "
            f"{source}"
        )

    source_fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
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

    if source_fps <= 0:
        source_fps = 30.0

    if (
        width <= 0
        or height <= 0
    ):
        capture.release()
        raise RuntimeError(
            "영상 해상도를 확인할 수 "
            f"없습니다: {source}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    option_name = (
        f"w{window_seconds:g}_"
        f"c{confidence_threshold:g}_"
        f"s{stabilization_buffer}"
    )

    output_video = (
        output_dir
        / (
            f"{source.stem}_"
            f"{option_name}.mp4"
        )
    )

    output_log = (
        output_dir
        / (
            f"{source.stem}_"
            f"{option_name}.csv"
        )
    )

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        source_fps,
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            "결과 영상을 저장할 수 "
            f"없습니다: {output_video}"
        )

    feature_buffer = (
        TimedFeatureBuffer(
            window_seconds
        )
    )

    stabilizer = (
        PredictionStabilizer(
            buffer_size=(
                stabilization_buffer
            ),
            min_votes=(
                stabilization_min_votes
            ),
        )
    )

    fps_meter = FPSMeter()

    last_prediction_time = (
        -1e9
    )

    raw_label = "WARMING_UP"
    threshold_label = (
        "WARMING_UP"
    )
    stable_label = (
        "WARMING_UP"
    )
    confidence = 0.0

    expected_label = (
        infer_expected_label(
            source
        )
    )

    frame_id = 0
    log_rows = []

    process_start = (
        time.perf_counter()
    )

    while True:
        success, frame = (
            capture.read()
        )

        if not success:
            break

        video_time = (
            frame_id
            / source_fps
        )

        loop_start = (
            time.perf_counter()
        )

        result = (
            pose_model.predict(
                source=frame,
                imgsz=int(
                    project_config[
                        "image_size"
                    ]
                ),
                conf=float(
                    project_config[
                        "person_confidence"
                    ]
                ),
                device=device,
                verbose=False,
            )[0]
        )

        feature = (
            frame_feature_from_pose(
                result
            )
        )

        feature_buffer.append(
            video_time,
            feature,
        )

        ready = (
            feature_buffer.is_ready(
                min_window_fill_ratio=float(
                    runtime_config[
                        "min_window_fill_ratio"
                    ]
                ),
                min_valid_feature_ratio=float(
                    runtime_config[
                        "min_valid_feature_ratio"
                    ]
                ),
                min_valid_features=int(
                    runtime_config[
                        "min_valid_features"
                    ]
                ),
            )
        )

        prediction_updated = 0

        if (
            ready
            and (
                video_time
                - last_prediction_time
            )
            >= float(
                runtime_config[
                    "prediction_interval_sec"
                ]
            )
        ):
            (
                threshold_label,
                raw_label,
                confidence,
                _,
            ) = predict_from_window(
                model=action_model,
                feature_buffer=(
                    feature_buffer
                ),
                sequence_steps=(
                    sequence_steps
                ),
                confidence_threshold=(
                    confidence_threshold
                ),
            )

            stable_label = (
                stabilizer.update(
                    threshold_label
                )
            )

            last_prediction_time = (
                video_time
            )

            prediction_updated = 1

        elif not ready:
            raw_label = (
                "WARMING_UP"
            )
            threshold_label = (
                "WARMING_UP"
            )
            stable_label = (
                "WARMING_UP"
            )
            confidence = 0.0

        annotated = (
            result.plot()
        )

        processing_fps = (
            fps_meter.update(
                time.perf_counter()
            )
        )

        cv2.rectangle(
            annotated,
            (10, 10),
            (690, 190),
            (0, 0, 0),
            -1,
        )

        texts = [
            (
                f"Expected: "
                f"{expected_label or '-'}"
            ),
            f"Raw: {raw_label}",
            (
                f"Threshold: "
                f"{threshold_label}"
            ),
            (
                f"Stable: "
                f"{stable_label}"
            ),
            (
                f"Conf: "
                f"{confidence:.2f}  "
                f"Window: "
                f"{window_seconds:.1f}s  "
                f"FPS: "
                f"{processing_fps:.1f}"
            ),
        ]

        for index, text in enumerate(
            texts
        ):
            cv2.putText(
                annotated,
                text,
                (
                    25,
                    38
                    + index * 34,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
            )

        writer.write(
            annotated
        )

        log_rows.append(
            {
                "frame_id": frame_id,
                "video_time_sec": round(
                    video_time,
                    4,
                ),
                "expected_label": (
                    expected_label
                ),
                "window_seconds": (
                    window_seconds
                ),
                "confidence_threshold": (
                    confidence_threshold
                ),
                "stabilization_buffer": (
                    stabilization_buffer
                ),
                "stabilization_min_votes": (
                    stabilization_min_votes
                ),
                "buffer_duration_sec": round(
                    feature_buffer.duration(),
                    4,
                ),
                "valid_feature_ratio": round(
                    feature_buffer.valid_ratio(),
                    4,
                ),
                "raw_label": raw_label,
                "threshold_label": (
                    threshold_label
                ),
                "stable_label": (
                    stable_label
                ),
                "confidence": round(
                    confidence,
                    4,
                ),
                "prediction_updated": (
                    prediction_updated
                ),
                "processing_fps": round(
                    processing_fps,
                    3,
                ),
                "frame_processing_sec": round(
                    time.perf_counter()
                    - loop_start,
                    5,
                ),
            }
        )

        frame_id += 1

        if frame_id % 30 == 0:
            print(
                f"Frame {frame_id} "
                f"Stable="
                f"{stable_label} "
                f"Conf="
                f"{confidence:.2f} "
                f"FPS="
                f"{processing_fps:.1f}"
            )

    capture.release()
    writer.release()

    if not log_rows:
        raise RuntimeError(
            "처리한 Frame이 없습니다."
        )

    total_elapsed = (
        time.perf_counter()
        - process_start
    )

    with output_log.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer_csv = (
            csv.DictWriter(
                file,
                fieldnames=list(
                    log_rows[0].keys()
                ),
            )
        )
        writer_csv.writeheader()
        writer_csv.writerows(
            log_rows
        )

    overall_fps = (
        frame_id
        / total_elapsed
        if total_elapsed > 0
        else 0.0
    )

    prediction_updates = sum(
        int(
            row[
                "prediction_updated"
            ]
        )
        for row in log_rows
    )

    print()
    print(
        f"Frames: {frame_id}"
    )
    print(
        "Prediction updates: "
        f"{prediction_updates}"
    )

    if prediction_updates == 0:
        print(
            "[WARNING] Window가 충분히 "
            "채워지기 전에 영상이 끝났습니다."
        )
        print(
            "더 긴 S02 영상 또는 더 짧은 "
            "Window로 다시 확인하세요."
        )
    print(
        "Overall processing FPS: "
        f"{overall_fps:.2f}"
    )
    print(
        f"Video: {output_video}"
    )
    print(
        f"Log  : {output_log}"
    )


if __name__ == "__main__":
    main()