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

sys.path.insert(0, str(ROOT))


from src.common import load_config, resolve_device

from src.multi_person_utils import (
    MultiPersonStateManager,
    feature_from_keypoints,
)

from src.realtime_utils import (
    FPSMeter,
    predict_from_window,
    validate_runtime_values,
)


DAY07_CONFIG_PATH = (
    ROOT / "configs" / "day07_realtime.json"
)

DAY08_CONFIG_PATH = (
    ROOT / "configs" / "day08_multi_person.json"
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
    ROOT / "reports" / "day08" / "ip_webcam"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        required=True,
        help=(
            "예: http://192.168.0.10:8080/video"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def track_color(track_id: int) -> tuple[int, int, int]:
    """Track ID마다 눈에 잘 구분되는 BGR 색상을 반환합니다."""
    palette = [
        (255, 170, 0),
        (0, 200, 255),
        (120, 220, 80),
        (220, 120, 255),
        (255, 120, 120),
        (80, 220, 220),
    ]
    return palette[int(track_id) % len(palette)]


def draw_badge(
    image,
    text: str,
    x: int,
    y: int,
    background_color,
    font_scale: float = 0.58,
    thickness: int = 2,
) -> None:
    """작은 배경 박스를 가진 텍스트 라벨을 그립니다."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    pad_x = 7
    pad_y = 5

    image_height, image_width = image.shape[:2]

    x = max(0, min(int(x), image_width - 1))
    y = max(0, min(int(y), image_height - 1))

    box_x1 = x
    box_y1 = y
    box_x2 = min(
        image_width - 1,
        box_x1 + text_width + pad_x * 2,
    )
    box_y2 = min(
        image_height - 1,
        box_y1 + text_height + baseline + pad_y * 2,
    )

    cv2.rectangle(
        image,
        (box_x1, box_y1),
        (box_x2, box_y2),
        background_color,
        -1,
    )

    text_x = box_x1 + pad_x
    text_y = min(
        image_height - 1,
        box_y1 + pad_y + text_height,
    )

    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_person_ui(
    image,
    bbox,
    track_id: int,
    stable_label: str,
    confidence: float,
) -> None:
    """
    한 사람의 BBox, ID, Action을 겹치지 않도록 표시합니다.

    - ID: BBox 왼쪽 위 내부
    - Action/Confidence: BBox 왼쪽 아래 내부
    """
    image_height, image_width = image.shape[:2]

    x1, y1, x2, y2 = [int(value) for value in bbox]

    x1 = max(0, min(x1, image_width - 1))
    x2 = max(0, min(x2, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    y2 = max(0, min(y2, image_height - 1))

    color = track_color(track_id)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        2,
        cv2.LINE_AA,
    )

    draw_badge(
        image,
        f"ID {track_id}",
        x1 + 4,
        y1 + 4,
        color,
        font_scale=0.62,
        thickness=2,
    )

    action_text = (
        f"{stable_label}  {confidence:.2f}"
    )

    action_y = max(
        y1 + 40,
        y2 - 34,
    )

    draw_badge(
        image,
        action_text,
        x1 + 4,
        action_y,
        (35, 35, 35),
        font_scale=0.50,
        thickness=1,
    )


def draw_status_panel(
    image,
    seen_ids: set[int],
    runtime_ids: list[int],
    fps: float,
    removed_ids: list[int],
) -> None:
    """화면 왼쪽 위에 최소한의 Runtime 상태만 표시합니다."""
    lines = [
        f"Seen: {sorted(seen_ids)}",
        f"States: {len(runtime_ids)}",
        f"FPS: {fps:.1f}",
    ]

    if removed_ids:
        lines.append(
            f"Removed: {removed_ids}"
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    line_height = 24
    padding = 10

    widths = [
        cv2.getTextSize(
            line,
            font,
            font_scale,
            thickness,
        )[0][0]
        for line in lines
    ]

    panel_width = max(widths) + padding * 2
    panel_height = (
        len(lines) * line_height
        + padding
    )

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (8, 8),
        (8 + panel_width, 8 + panel_height),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.58,
        image,
        0.42,
        0,
        image,
    )

    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (
                18,
                29 + index * line_height,
            ),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def draw_controls(image) -> None:
    """화면 왼쪽 아래에 조작키를 작게 표시합니다."""
    text = "R: reset   Q: quit"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    height, _ = image.shape[:2]
    x = 12
    y = height - 12

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (
            x - 5,
            y - text_height - 8,
        ),
        (
            x + text_width + 7,
            y + baseline + 4,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.55,
        image,
        0.45,
        0,
        image,
    )

    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    project_config = load_config()
    day07_config = load_json(DAY07_CONFIG_PATH)
    day08_config = load_json(DAY08_CONFIG_PATH)
    training_info = load_json(TRAIN_INFO_PATH)

    sequence_steps = int(training_info["sequence_steps"])

    window_seconds = float(
        day07_config["window_seconds"]
    )
    confidence_threshold = float(
        day07_config["confidence_threshold"]
    )
    stabilization_buffer = int(
        day07_config["stabilization_buffer"]
    )
    stabilization_min_votes = int(
        day07_config["stabilization_min_votes"]
    )

    validate_runtime_values(
        window_seconds=window_seconds,
        confidence_threshold=confidence_threshold,
        stabilization_buffer=stabilization_buffer,
        stabilization_min_votes=(
            stabilization_min_votes
        ),
    )

    device = resolve_device(
        project_config["device"]
    )

    action_model = joblib.load(MODEL_PATH)

    expected_features = sequence_steps * 38

    if int(action_model.n_features_in_) != expected_features:
        raise RuntimeError(
            "6일차 모델 Feature 수와 "
            "현재 sequence_steps가 다릅니다."
        )

    pose_model = YOLO(project_config["pose_model"])

    capture = cv2.VideoCapture(args.url)

    if not capture.isOpened():
        raise RuntimeError(
            "IP Webcam Stream을 열 수 없습니다."
        )

    success, frame = capture.read()

    if not success or frame is None:
        capture.release()
        raise RuntimeError(
            "IP Webcam 첫 Frame을 읽지 못했습니다."
        )

    height, width = frame.shape[:2]

    print(f"Connected: {width} x {height}")
    print("Q: quit")
    print("R: reset all person states")
    print()

    state_manager = MultiPersonStateManager(
        window_seconds=window_seconds,
        stabilization_buffer=stabilization_buffer,
        stabilization_min_votes=(
            stabilization_min_votes
        ),
        lost_timeout_seconds=float(
            day08_config["lost_timeout_seconds"]
        ),
    )

    fps_meter = FPSMeter()
    session_start = time.perf_counter()
    frame_id = 0
    log_rows = []

    while True:
        elapsed = time.perf_counter() - session_start

        result = pose_model.track(
            source=frame,
            persist=True,
            tracker=day08_config["tracker"],
            imgsz=int(project_config["image_size"]),
            conf=float(
                project_config["person_confidence"]
            ),
            device=device,
            verbose=False,
        )[0]

        # Ultralytics 기본 BBox/라벨은 숨기고
        # Skeleton만 그립니다.
        # 아래에서 Track ID와 Action 라벨을 직접 그려
        # 글자 중복을 방지합니다.
        annotated = result.plot(
            boxes=False,
            labels=False,
        )
        current_fps = fps_meter.update(
            time.perf_counter()
        )

        boxes = result.boxes
        keypoints = result.keypoints
        seen_ids: set[int] = set()

        if (
            boxes is not None
            and boxes.id is not None
            and keypoints is not None
            and keypoints.xy is not None
        ):
            track_ids = (
                boxes.id.detach().cpu().numpy().astype(int)
            )
            xyxy = boxes.xyxy.detach().cpu().numpy()
            kp_xy = (
                keypoints.xy.detach().cpu().numpy()
            )

            item_count = min(
                len(track_ids),
                len(xyxy),
                len(kp_xy),
            )

            for index in range(item_count):
                track_id = int(track_ids[index])
                seen_ids.add(track_id)

                state = state_manager.get_or_create(
                    track_id,
                    elapsed,
                )

                feature = feature_from_keypoints(
                    kp_xy[index]
                )

                state.feature_buffer.append(
                    elapsed,
                    feature,
                )

                ready = state.feature_buffer.is_ready(
                    min_window_fill_ratio=float(
                        day07_config[
                            "min_window_fill_ratio"
                        ]
                    ),
                    min_valid_feature_ratio=float(
                        day07_config[
                            "min_valid_feature_ratio"
                        ]
                    ),
                    min_valid_features=int(
                        day07_config[
                            "min_valid_features"
                        ]
                    ),
                )

                prediction_updated = 0

                if (
                    ready
                    and (
                        elapsed
                        - state.last_prediction_time
                    )
                    >= float(
                        day07_config[
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
                            state.feature_buffer
                        ),
                        sequence_steps=sequence_steps,
                        confidence_threshold=(
                            confidence_threshold
                        ),
                    )

                    state.raw_label = raw_label
                    state.threshold_label = (
                        threshold_label
                    )
                    state.confidence = confidence
                    state.stable_label = (
                        state.stabilizer.update(
                            threshold_label
                        )
                    )
                    state.last_prediction_time = elapsed
                    prediction_updated = 1

                elif not ready:
                    state.raw_label = "WARMING_UP"
                    state.threshold_label = "WARMING_UP"
                    state.stable_label = "WARMING_UP"
                    state.confidence = 0.0

                x1, y1, x2, y2 = xyxy[index]

                # 사람마다 한 개의 BBox만 그리고,
                # ID와 Action을 위/아래로 분리해 표시합니다.
                draw_person_ui(
                    annotated,
                    (x1, y1, x2, y2),
                    track_id,
                    state.stable_label,
                    state.confidence,
                )

                log_rows.append(
                    {
                        "frame_id": frame_id,
                        "elapsed_sec": round(
                            elapsed,
                            4,
                        ),
                        "track_id": track_id,
                        "buffer_duration_sec": round(
                            state.feature_buffer.duration(),
                            4,
                        ),
                        "valid_feature_ratio": round(
                            state.feature_buffer.valid_ratio(),
                            4,
                        ),
                        "raw_label": state.raw_label,
                        "threshold_label": (
                            state.threshold_label
                        ),
                        "stable_label": (
                            state.stable_label
                        ),
                        "confidence": round(
                            state.confidence,
                            4,
                        ),
                        "prediction_updated": (
                            prediction_updated
                        ),
                        "processing_fps": round(
                            current_fps,
                            3,
                        ),
                    }
                )

        state_manager.mark_missing(
            elapsed,
            seen_ids,
        )

        removed_ids = state_manager.remove_lost(
            elapsed
        )

        # 상태 정보는 반투명 패널 하나에 간단히 표시합니다.
        draw_status_panel(
            annotated,
            seen_ids,
            state_manager.active_ids(),
            current_fps,
            removed_ids,
        )

        draw_controls(annotated)

        cv2.imshow(
            "Subject10 Multi-Person Action",
            annotated,
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        if key in (ord("r"), ord("R")):
            state_manager.clear()
            print("All person runtime states reset")

        success, frame = capture.read()

        if not success or frame is None:
            print("다음 IP Webcam Frame을 읽지 못했습니다.")
            break

        frame_id += 1

    capture.release()
    cv2.destroyAllWindows()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_log = (
        output_dir
        / f"ip_multi_{timestamp}.csv"
    )

    if log_rows:
        with output_log.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(log_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(log_rows)

        print()
        print(f"Saved: {output_log}")
    else:
        print()
        print(
            "Track ID가 있는 Runtime Log가 없습니다. "
            "두 사람이 충분히 보이는지 확인하세요."
        )


if __name__ == "__main__":
    main()