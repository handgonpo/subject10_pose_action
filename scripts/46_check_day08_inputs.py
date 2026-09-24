from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import joblib


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


from src.realtime_utils import (
    validate_runtime_values,
)
from src.sequence_utils import (
    sequence_feature_names,
)


VIDEO_DIR = ROOT / "data" / "day08" / "videos"

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

DAY07_CONFIG_PATH = (
    ROOT
    / "configs"
    / "day07_realtime.json"
)

DAY08_CONFIG_PATH = (
    ROOT
    / "configs"
    / "day08_multi_person.json"
)

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day08"
    / "tables"
    / "day08_input_check.csv"
)

EXPECTED_VIDEOS = {
    "01_sbu_shaking_hands.mp4": "BASIC_TRACKING",
    "02_sbu_hugging.mp4": "OCCLUSION_TRACKING",
    "03_sbu_kicking_challenge.mp4": "FAST_INTERACTION_TRACKING",
}

EXPECTED_ACTIONS = {
    "bend_return",
    "leg_raise_lower",
    "walk_turn_walk",
    "drink_return",
    "sit_stand",
    "wave",
}

REQUIRED_TRAIN_INFO_KEYS = {
    "classes",
    "feature_count",
    "sequence_steps",
}

REQUIRED_DAY07_KEYS = {
    "window_seconds",
    "min_window_fill_ratio",
    "min_valid_feature_ratio",
    "min_valid_features",
    "prediction_interval_sec",
    "confidence_threshold",
    "stabilization_buffer",
    "stabilization_min_votes",
}


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_day07_config(
    config: dict,
) -> None:
    missing = (
        REQUIRED_DAY07_KEYS
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

    validate_runtime_values(
        window_seconds=float(
            config["window_seconds"]
        ),
        confidence_threshold=float(
            config[
                "confidence_threshold"
            ]
        ),
        stabilization_buffer=int(
            config[
                "stabilization_buffer"
            ]
        ),
        stabilization_min_votes=int(
            config[
                "stabilization_min_votes"
            ]
        ),
    )

    if not (
        0
        < float(
            config[
                "min_window_fill_ratio"
            ]
        )
        <= 1
    ):
        raise RuntimeError(
            "min_window_fill_ratio는 "
            "0 초과 1 이하여야 합니다."
        )

    if not (
        0
        < float(
            config[
                "min_valid_feature_ratio"
            ]
        )
        <= 1
    ):
        raise RuntimeError(
            "min_valid_feature_ratio는 "
            "0 초과 1 이하여야 합니다."
        )

    if int(
        config[
            "min_valid_features"
        ]
    ) < 2:
        raise RuntimeError(
            "min_valid_features는 "
            "2 이상이어야 합니다."
        )

    if float(
        config[
            "prediction_interval_sec"
        ]
    ) <= 0:
        raise RuntimeError(
            "prediction_interval_sec는 "
            "0보다 커야 합니다."
        )


def read_video_info(path: Path) -> dict:
    capture = cv2.VideoCapture(
        str(path)
    )

    if not capture.isOpened():
        return {
            "open_ok": 0,
            "fps": 0.0,
            "frames": 0,
            "width": 0,
            "height": 0,
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

    success, frame = capture.read()
    capture.release()

    open_ok = int(
        success
        and frame is not None
        and fps > 0
        and frames > 0
        and width > 0
        and height > 0
    )

    duration_sec = (
        frames / fps
        if fps > 0
        else 0.0
    )

    return {
        "open_ok": open_ok,
        "fps": round(
            fps,
            3,
        ),
        "frames": frames,
        "width": width,
        "height": height,
        "duration_sec": round(
            duration_sec,
            3,
        ),
    }


def main() -> None:
    required_files = [
        MODEL_PATH,
        FEATURE_LIST_PATH,
        TRAIN_INFO_PATH,
        DAY07_CONFIG_PATH,
        DAY08_CONFIG_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    training_info = load_json(
        TRAIN_INFO_PATH
    )
    feature_columns = load_json(
        FEATURE_LIST_PATH
    )
    day07_config = load_json(
        DAY07_CONFIG_PATH
    )
    day08_config = load_json(
        DAY08_CONFIG_PATH
    )
    model = joblib.load(
        MODEL_PATH
    )

    missing_train_info = (
        REQUIRED_TRAIN_INFO_KEYS
        - set(training_info)
    )

    if missing_train_info:
        raise RuntimeError(
            "rf_training_info.json에 "
            "필수 정보가 없습니다: "
            + ", ".join(
                sorted(
                    missing_train_info
                )
            )
        )

    validate_day07_config(
        day07_config
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

    expected_feature_count = (
        sequence_steps
        * 38
    )

    training_feature_count = int(
        training_info[
            "feature_count"
        ]
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

    expected_columns = (
        sequence_feature_names(
            sequence_steps
        )
    )

    if (
        feature_columns
        != expected_columns
    ):
        raise RuntimeError(
            "6일차 Feature Column 순서와 "
            "현재 sequence_utils.py가 "
            "다릅니다."
        )

    if (
        len(feature_columns)
        != expected_feature_count
    ):
        raise RuntimeError(
            "Feature Column 수가 "
            "sequence_steps × 38과 "
            "다릅니다."
        )

    model_feature_count = int(
        getattr(
            model,
            "n_features_in_",
            -1,
        )
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
        != EXPECTED_ACTIONS
        or training_classes
        != EXPECTED_ACTIONS
    ):
        raise RuntimeError(
            "6일차 Action Class가 "
            "현재 A001~A006과 다릅니다."
        )

    if (
        day08_config.get(
            "tracker"
        )
        != "bytetrack.yaml"
    ):
        raise RuntimeError(
            "8일차 기본 실습 Tracker는 "
            "bytetrack.yaml입니다."
        )

    if float(
        day08_config.get(
            "lost_timeout_seconds",
            0,
        )
    ) <= 0:
        raise RuntimeError(
            "lost_timeout_seconds는 "
            "0보다 커야 합니다."
        )

    rows = []

    for filename, role in (
        EXPECTED_VIDEOS.items()
    ):
        path = (
            VIDEO_DIR
            / filename
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        info = read_video_info(
            path
        )

        if info["open_ok"] != 1:
            raise RuntimeError(
                "영상을 정상적으로 "
                f"읽을 수 없습니다: {path}"
            )

        rows.append(
            {
                "video": filename,
                "role": role,
                "fps": info["fps"],
                "frames": (
                    info["frames"]
                ),
                "width": (
                    info["width"]
                ),
                "height": (
                    info["height"]
                ),
                "duration_sec": (
                    info[
                        "duration_sec"
                    ]
                ),
                "use_for_tracking": 1,
                "use_as_rf_ground_truth": 0,
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
        writer.writerows(
            rows
        )

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
        "SBU videos     : 3 / 3"
    )
    print(
        "Tracker        : "
        "bytetrack.yaml"
    )
    print()

    for row in rows:
        print(
            f"{row['video']:34s} "
            f"frames={row['frames']:3d} "
            f"fps={row['fps']:.3f} "
            f"duration="
            f"{row['duration_sec']:.3f}s"
        )

    print()
    print(
        f"Saved: "
        f"{OUTPUT_PATH}"
    )
    print(
        "Day08 input data: READY"
    )


if __name__ == "__main__":
    main()