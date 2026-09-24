from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import joblib


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from src.sequence_utils import (
    sequence_feature_names,
)


MODEL_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_action_baseline.joblib"
)

FEATURE_LIST_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_feature_columns.json"
)

TRAIN_INFO_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_training_info.json"
)

PROJECT_CONFIG_PATH = (
    ROOT
    / "configs"
    / "project.json"
)

DAY07_CONFIG_PATH = (
    ROOT
    / "configs"
    / "day07_realtime.json"
)

VIDEO_ROOT = (
    ROOT
    / "data"
    / "day05"
    / "videos"
    / "S02"
)

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day07"
    / "tables"
    / "day07_input_check.csv"
)

EXPECTED_ACTIONS = {
    "A001": "bend_return",
    "A002": "leg_raise_lower",
    "A003": "walk_turn_walk",
    "A004": "drink_return",
    "A005": "sit_stand",
    "A006": "wave",
}

REQUIRED_TRAIN_INFO_KEYS = {
    "classes",
    "feature_count",
    "sequence_steps",
}


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_runtime_config(
    config: dict,
) -> None:
    required = {
        "window_seconds",
        "min_window_fill_ratio",
        "min_valid_feature_ratio",
        "min_valid_features",
        "prediction_interval_sec",
        "confidence_threshold",
        "stabilization_buffer",
        "stabilization_min_votes",
    }

    missing = (
        required
        - set(config)
    )

    if missing:
        raise RuntimeError(
            "day07_realtime.json에 "
            "필수 설정이 없습니다: "
            + ", ".join(
                sorted(missing)
            )
        )

    window_seconds = float(
        config["window_seconds"]
    )
    fill_ratio = float(
        config[
            "min_window_fill_ratio"
        ]
    )
    valid_ratio = float(
        config[
            "min_valid_feature_ratio"
        ]
    )
    min_valid_features = int(
        config[
            "min_valid_features"
        ]
    )
    prediction_interval = float(
        config[
            "prediction_interval_sec"
        ]
    )
    confidence = float(
        config[
            "confidence_threshold"
        ]
    )
    buffer_size = int(
        config[
            "stabilization_buffer"
        ]
    )
    min_votes = int(
        config[
            "stabilization_min_votes"
        ]
    )

    if window_seconds <= 0:
        raise RuntimeError(
            "window_seconds는 "
            "0보다 커야 합니다."
        )

    if not (
        0 < fill_ratio <= 1
    ):
        raise RuntimeError(
            "min_window_fill_ratio는 "
            "0 초과 1 이하여야 합니다."
        )

    if not (
        0 < valid_ratio <= 1
    ):
        raise RuntimeError(
            "min_valid_feature_ratio는 "
            "0 초과 1 이하여야 합니다."
        )

    if min_valid_features < 2:
        raise RuntimeError(
            "min_valid_features는 "
            "2 이상이어야 합니다."
        )

    if prediction_interval <= 0:
        raise RuntimeError(
            "prediction_interval_sec는 "
            "0보다 커야 합니다."
        )

    if not (
        0 <= confidence <= 1
    ):
        raise RuntimeError(
            "confidence_threshold는 "
            "0~1 범위여야 합니다."
        )

    if buffer_size < 1:
        raise RuntimeError(
            "stabilization_buffer는 "
            "1 이상이어야 합니다."
        )

    if not (
        1
        <= min_votes
        <= buffer_size
    ):
        raise RuntimeError(
            "stabilization_min_votes는 "
            "1 이상이고 "
            "stabilization_buffer보다 "
            "클 수 없습니다."
        )


def read_video_info(
    path: Path,
) -> dict:
    capture = cv2.VideoCapture(
        str(path)
    )

    if not capture.isOpened():
        return {
            "open_ok": 0,
            "fps": 0.0,
            "frames": 0,
            "duration_sec": 0.0,
        }

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )
    frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    success, frame = capture.read()
    capture.release()

    if (
        not success
        or frame is None
        or fps <= 0
        or frames <= 0
    ):
        return {
            "open_ok": 0,
            "fps": fps,
            "frames": frames,
            "duration_sec": 0.0,
        }

    duration = (
        frames / fps
    )

    return {
        "open_ok": 1,
        "fps": round(
            fps,
            3,
        ),
        "frames": frames,
        "duration_sec": round(
            duration,
            3,
        ),
    }


def main() -> None:
    required_files = [
        MODEL_PATH,
        FEATURE_LIST_PATH,
        TRAIN_INFO_PATH,
        PROJECT_CONFIG_PATH,
        DAY07_CONFIG_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    model = joblib.load(
        MODEL_PATH
    )

    feature_columns = load_json(
        FEATURE_LIST_PATH
    )

    training_info = load_json(
        TRAIN_INFO_PATH
    )

    runtime_config = load_json(
        DAY07_CONFIG_PATH
    )

    missing_train_info = (
        REQUIRED_TRAIN_INFO_KEYS
        - set(training_info)
    )

    if missing_train_info:
        raise RuntimeError(
            "rf_training_info.json에 "
            "7일차에 필요한 정보가 없습니다: "
            + ", ".join(
                sorted(
                    missing_train_info
                )
            )
            + "\n6일차 최종 수정본의 "
            "35_train_rf_baseline.py를 "
            "다시 실행하세요."
        )

    validate_runtime_config(
        runtime_config
    )

    sequence_steps = int(
        training_info[
            "sequence_steps"
        ]
    )

    if sequence_steps < 2:
        raise RuntimeError(
            "sequence_steps는 "
            "2 이상이어야 합니다."
        )

    expected_feature_columns = (
        sequence_feature_names(
            sequence_steps
        )
    )

    if (
        feature_columns
        != expected_feature_columns
    ):
        raise RuntimeError(
            "6일차 모델의 Feature Column과 "
            "현재 sequence_utils.py의 "
            "Feature 순서가 다릅니다."
        )

    expected_feature_count = (
        sequence_steps
        * 38
    )

    training_feature_count = int(
        training_info[
            "feature_count"
        ]
    )

    model_feature_count = int(
        getattr(
            model,
            "n_features_in_",
            -1,
        )
    )

    if (
        len(feature_columns)
        != expected_feature_count
    ):
        raise RuntimeError(
            "Feature Column 수가 "
            f"{expected_feature_count}개가 "
            f"아닙니다."
        )

    if (
        training_feature_count
        != expected_feature_count
    ):
        raise RuntimeError(
            "rf_training_info.json의 "
            "feature_count가 "
            "sequence_steps × 38과 "
            "다릅니다."
        )

    if (
        model_feature_count
        != expected_feature_count
    ):
        raise RuntimeError(
            "Random Forest 입력 Feature 수가 "
            "6일차 설정과 다릅니다."
        )

    model_classes = set(
        map(
            str,
            model.classes_,
        )
    )

    expected_classes = set(
        EXPECTED_ACTIONS.values()
    )

    training_classes = set(
        map(
            str,
            training_info[
                "classes"
            ],
        )
    )

    if (
        model_classes
        != expected_classes
        or training_classes
        != expected_classes
    ):
        raise RuntimeError(
            "6일차 모델 또는 Training Info의 "
            "Action Class가 A001~A006과 "
            "일치하지 않습니다."
        )

    required_duration = (
        float(
            runtime_config[
                "window_seconds"
            ]
        )
        * float(
            runtime_config[
                "min_window_fill_ratio"
            ]
        )
    )

    rows = []
    runtime_ready_count = 0

    for action_id, action_label in (
        EXPECTED_ACTIONS.items()
    ):
        video_path = (
            VIDEO_ROOT
            / (
                f"S02_{action_id}_"
                "T01.mp4"
            )
        )

        if not video_path.exists():
            raise FileNotFoundError(
                video_path
            )

        info = read_video_info(
            video_path
        )

        if info["open_ok"] != 1:
            raise RuntimeError(
                "S02 영상을 정상적으로 "
                "읽을 수 없습니다: "
                f"{video_path}"
            )

        runtime_ready = int(
            info[
                "duration_sec"
            ]
            >= required_duration
        )

        runtime_ready_count += (
            runtime_ready
        )

        rows.append(
            {
                "action_id": action_id,
                "action_label": (
                    action_label
                ),
                "video_path": str(
                    video_path
                    .relative_to(ROOT)
                ),
                "exists": 1,
                "fps": info["fps"],
                "frames": (
                    info["frames"]
                ),
                "duration_sec": (
                    info[
                        "duration_sec"
                    ]
                ),
                "required_warmup_sec": (
                    round(
                        required_duration,
                        3,
                    )
                ),
                "runtime_ready": (
                    runtime_ready
                ),
            }
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Sequence steps : "
        f"{sequence_steps}"
    )
    print(
        f"Feature count  : "
        f"{expected_feature_count}"
    )
    print(
        "Action classes :",
        sorted(
            model_classes
        ),
    )
    print(
        "S02 videos     : 6 / 6"
    )
    print(
        "Runtime-ready  : "
        f"{runtime_ready_count} / 6"
    )

    if runtime_ready_count < 6:
        print()
        print(
            "[WARNING] 일부 S02 영상은 "
            "현재 Window의 Warm-up 시간보다 "
            "짧습니다."
        )
        print(
            "day07_input_check.csv에서 "
            "runtime_ready=1인 영상을 "
            "저장영상 첫 실습에 사용하세요."
        )

    print()
    print(
        f"Saved: "
        f"{OUTPUT_PATH}"
    )
    print(
        "Day07 input data: READY"
    )


if __name__ == "__main__":
    main()