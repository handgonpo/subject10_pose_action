from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
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
    / "ip_webcam"
)

KEY_TO_ACTION = {
    ord("1"): "bend_return",
    ord("2"): "leg_raise_lower",
    ord("3"): "walk_turn_walk",
    ord("4"): "drink_return",
    ord("5"): "sit_stand",
    ord("6"): "wave",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        required=True,
        help=(
            "예: "
            "http://192.168.0.10:"
            "8080/video"
        ),
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
        "--keep-buffer-on-target",
        action="store_true",
        help=(
            "Target 키를 바꿀 때 "
            "Sliding Window를 유지하여 "
            "연속 Action 전환을 관찰합니다."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
        help=(
            "Runtime Log를 "
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


def read_frame_with_retry(
    capture: cv2.VideoCapture,
    max_retries: int = 10,
    retry_delay_sec: float = 0.05,
):
    """
    IP Webcam의 MJPEG Stream에서 한 번의 일시적인 read 실패가
    발생해도 바로 종료하지 않고 짧게 재시도합니다.
    """
    for attempt in range(1, max_retries + 1):
        success, frame = capture.read()

        if success and frame is not None:
            return True, frame

        if attempt < max_retries:
            time.sleep(retry_delay_sec)

    return False, None


def main() -> None:
    args = parse_args()

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

    keep_buffer_on_target = bool(
        args.keep_buffer_on_target
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
        args.url
    )

    if not capture.isOpened():
        raise RuntimeError(
            "IP Webcam Stream을 "
            "열 수 없습니다."
        )

    success, frame = (
        read_frame_with_retry(
            capture
        )
    )

    if (
        not success
        or frame is None
    ):
        capture.release()
        raise RuntimeError(
            "IP Webcam 첫 Frame을 "
            "읽지 못했습니다."
        )

    height, width = (
        frame.shape[:2]
    )

    WINDOW_NAME = "Subject10 IP Webcam Action"
    PANEL_WIDTH = 380

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        width + PANEL_WIDTH,
        height,
    )

    print(
        f"Connected: "
        f"{width} x {height}"
    )
    print()
    print(
        "1 bend_return"
    )
    print(
        "2 leg_raise_lower"
    )
    print(
        "3 walk_turn_walk"
    )
    print(
        "4 drink_return"
    )
    print(
        "5 sit_stand"
    )
    print(
        "6 wave"
    )
    print(
        "0 target clear"
    )
    print(
        "R buffer reset"
    )
    print(
        "Q quit"
    )
    print()
    print(
        "Target mode:",
        (
            "KEEP BUFFER"
            if keep_buffer_on_target
            else "RESET BUFFER"
        ),
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
    expected_label = ""

    session_start = (
        time.perf_counter()
    )

    log_rows = []
    frame_id = 0
    segment_id = 0

    while True:
        now = time.perf_counter()
        elapsed = (
            now
            - session_start
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
            elapsed,
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
                elapsed
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
                elapsed
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

        # -------------------------------------------------
        # 카메라 영상 오른쪽에 Runtime 정보 Panel을 붙입니다.
        # 카메라 원본에는 검은 박스나 텍스트를 덮지 않으므로
        # 사람의 머리부터 발끝까지 Pose를 그대로 확인할 수 있습니다.
        # -------------------------------------------------
        display_frame = cv2.copyMakeBorder(
            annotated,
            0,
            0,
            0,
            PANEL_WIDTH,
            cv2.BORDER_CONSTANT,
            value=(25, 25, 25),
        )

        panel_x = width + 20

        cv2.putText(
            display_frame,
            "RUNTIME STATUS",
            (panel_x, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )

        texts = [
            f"Target: {expected_label or '-'}",
            f"Raw: {raw_label}",
            f"Threshold: {threshold_label}",
            f"Stable: {stable_label}",
            f"Confidence: {confidence:.2f}",
            f"Window: {window_seconds:.1f}s",
            f"FPS: {processing_fps:.1f}",
            (
                "Mode: "
                + (
                    "KEEP"
                    if keep_buffer_on_target
                    else "RESET"
                )
            ),
        ]

        start_y = 72
        line_gap = 32

        for index, text in enumerate(
            texts
        ):
            cv2.putText(
                display_frame,
                text,
                (
                    panel_x,
                    start_y
                    + index * line_gap,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        separator_y = 340

        cv2.line(
            display_frame,
            (
                width + 10,
                separator_y,
            ),
            (
                width
                + PANEL_WIDTH
                - 10,
                separator_y,
            ),
            (120, 120, 120),
            1,
        )

        controls = [
            "1-6 : Target",
            "0   : Clear",
            "R   : Reset",
            "Q   : Quit",
        ]

        for index, text in enumerate(
            controls
        ):
            cv2.putText(
                display_frame,
                text,
                (
                    panel_x,
                    370 + index * 24,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow(
            WINDOW_NAME,
            display_frame,
        )

        log_rows.append(
            {
                "frame_id": frame_id,
                "elapsed_sec": round(
                    elapsed,
                    4,
                ),
                "expected_label": (
                    expected_label
                ),
                "segment_id": (
                    segment_id
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
                "keep_buffer_on_target": int(
                    keep_buffer_on_target
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
            }
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key in (
            ord("q"),
            ord("Q"),
        ):
            break

        if key in KEY_TO_ACTION:
            expected_label = (
                KEY_TO_ACTION[key]
            )
            segment_id += 1

            if not keep_buffer_on_target:
                feature_buffer.clear()
                stabilizer.clear()
                last_prediction_time = (
                    -1e9
                )
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

            print(
                "Target:",
                expected_label,
                "| segment:",
                segment_id,
                "| buffer:",
                (
                    "KEEP"
                    if keep_buffer_on_target
                    else "RESET"
                ),
            )

        elif key == ord("0"):
            expected_label = ""
            segment_id += 1
            feature_buffer.clear()
            stabilizer.clear()
            last_prediction_time = (
                -1e9
            )
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

            print(
                "Target cleared"
            )

        elif key in (
            ord("r"),
            ord("R"),
        ):
            feature_buffer.clear()
            stabilizer.clear()
            last_prediction_time = (
                -1e9
            )
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

            if expected_label:
                segment_id += 1

            print(
                "Buffers reset"
            )

        success, frame = (
            read_frame_with_retry(
                capture
            )
        )

        if (
            not success
            or frame is None
        ):
            print(
                "IP Webcam Frame을 여러 번 "
                "재시도했지만 읽지 못했습니다."
            )
            break

        frame_id += 1

    capture.release()
    cv2.destroyAllWindows()

    if not log_rows:
        raise RuntimeError(
            "저장할 Runtime Log가 "
            "없습니다."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    option_name = (
        f"w{window_seconds:g}_"
        f"c{confidence_threshold:g}_"
        f"s{stabilization_buffer}"
    )

    output_log = (
        output_dir
        / (
            "ip_webcam_"
            f"{option_name}_"
            f"{timestamp}.csv"
        )
    )

    with output_log.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                log_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            log_rows
        )

    fps_values = [
        float(
            row[
                "processing_fps"
            ]
        )
        for row in log_rows
        if float(
            row[
                "processing_fps"
            ]
        ) > 0
    ]

    mean_fps = (
        sum(fps_values)
        / len(fps_values)
        if fps_values
        else 0.0
    )

    print()
    print(
        "Mean processing FPS: "
        f"{mean_fps:.2f}"
    )
    print(
        f"Saved: {output_log}"
    )


if __name__ == "__main__":
    main()