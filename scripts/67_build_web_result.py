from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


# ------------------------------------------------------------
# 프로젝트 Root
# ------------------------------------------------------------

ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from src.common import (
    load_config,
    resolve_device,
)


# ------------------------------------------------------------
# 기존 Runtime 코드
# ------------------------------------------------------------

RUNTIME_SCRIPT = (
    ROOT
    / "scripts"
    / "42_realtime_action_video.py"
)


# ------------------------------------------------------------
# PART 7 결과 폴더
# ------------------------------------------------------------

WEB_DIR = (
    ROOT
    / "reports"
    / "day10"
    / "web"
)

JOB_DIR = (
    WEB_DIR
    / "jobs"
)

RUNTIME_DIR = (
    WEB_DIR
    / "runtime"
)

LATEST_PATH = (
    WEB_DIR
    / "latest_result.json"
)

HISTORY_PATH = (
    WEB_DIR
    / "prediction_history.json"
)


# ------------------------------------------------------------
# PART 4~6에서 확정한 Runtime 조건
# ------------------------------------------------------------

WINDOW_SECONDS = 3.0

CONFIDENCE_THRESHOLD = 0.20

STABILIZATION_BUFFER = 5


# ------------------------------------------------------------
# 현재 모델이 분류할 수 있는 6개 Action
# ------------------------------------------------------------

ACTION_INFO = {
    "bend_return":
        "허리를 굽혔다 다시 편다",

    "leg_raise_lower":
        "한쪽 다리를 들었다 내린다",

    "walk_turn_walk":
        "걷다가 돌아서 다시 걷는다",

    "drink_return":
        "음료수를 마시고 다시 앞을 본다",

    "sit_stand":
        "의자에 앉았다 다시 일어선다",

    "wave":
        "한 손을 들어 흔든다",
}


# ------------------------------------------------------------
# 실행 인자
# ------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help="분석할 MP4 파일",
    )

    parser.add_argument(
        "--expected",
        required=True,
        choices=sorted(
            ACTION_INFO
        ),
        help=(
            "영상에서 실제로 "
            "수행한 Action"
        ),
    )

    parser.add_argument(
        "--job-id",
        required=True,
        help="Web 분석 작업 ID",
    )

    return parser.parse_args()


# ------------------------------------------------------------
# 상대경로 → 절대경로
# ------------------------------------------------------------

def resolve_path(
    value: str,
) -> Path:

    path = Path(
        value
    )

    if not path.is_absolute():

        path = (
            ROOT
            / path
        )

    return path.resolve()


# ------------------------------------------------------------
# job_id 검사
# ------------------------------------------------------------

def validate_job_id(
    job_id: str,
) -> None:

    safe_value = (
        job_id.replace(
            "_",
            "",
        )
    )

    if not safe_value.isalnum():

        raise ValueError(
            "job-id에는 "
            "영문, 숫자, 밑줄만 "
            "사용할 수 있습니다."
        )


# ------------------------------------------------------------
# prediction_updated 값 확인
# ------------------------------------------------------------

def prediction_mask(
    series: pd.Series,
) -> pd.Series:

    if pd.api.types.is_bool_dtype(
        series
    ):

        return series.fillna(
            False
        )

    values = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return values.isin(
        {
            "1",
            "1.0",
            "true",
            "yes",
        }
    )


# ------------------------------------------------------------
# 0보다 큰 FPS 평균
# ------------------------------------------------------------

def positive_mean(
    series: pd.Series,
):

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values[
        values > 0
    ]

    if values.empty:

        return None

    return round(
        float(
            values.mean()
        ),
        3,
    )


# ------------------------------------------------------------
# Web에 표시할 Pose 대표 Frame 생성
# ------------------------------------------------------------

def write_pose_preview(
    source_path: Path,
    preview_path: Path,
    video_time_sec: float | None,
) -> None:

    capture = (
        cv2.VideoCapture(
            str(
                source_path
            )
        )
    )

    if not capture.isOpened():

        raise RuntimeError(
            "원본 영상을 "
            "열 수 없습니다: "
            f"{source_path}"
        )

    source_fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    frame_count = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    # Prediction이 하나도 없으면
    # 영상 중간 Frame을 사용합니다.

    if (
        video_time_sec is None
        and source_fps > 0
        and frame_count > 0
    ):

        video_time_sec = (
            frame_count
            / source_fps
            / 2.0
        )


    # 마지막 Prediction이 만들어진
    # 시간으로 이동합니다.

    if video_time_sec is not None:

        capture.set(
            cv2.CAP_PROP_POS_MSEC,
            max(
                float(
                    video_time_sec
                ),
                0.0,
            )
            * 1000.0,
        )


    success, frame = (
        capture.read()
    )


    # 해당 위치의 Frame을 읽지 못하면
    # 첫 번째 Frame으로 다시 시도합니다.

    if (
        not success
        or frame is None
    ):

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            0,
        )

        success, frame = (
            capture.read()
        )


    capture.release()


    if (
        not success
        or frame is None
    ):

        raise RuntimeError(
            "대표 Frame을 "
            "읽을 수 없습니다."
        )


    # --------------------------------------------------------
    # 현재 프로젝트의 Pose 설정을 사용합니다.
    # --------------------------------------------------------

    project_config = (
        load_config()
    )

    device = (
        resolve_device(
            project_config[
                "device"
            ]
        )
    )

    pose_model = YOLO(
        project_config[
            "pose_model"
        ]
    )


    pose_result = (
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


    # YOLO Pose 결과를 이용하여
    # 사람 Bounding Box와 Skeleton을 그립니다.

    annotated = (
        pose_result.plot()
    )


    preview_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    if not cv2.imwrite(
        str(
            preview_path
        ),
        annotated,
    ):

        raise RuntimeError(
            "preview.jpg를 "
            "저장할 수 없습니다."
        )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:

    args = parse_args()


    # --------------------------------------------------------
    # 1. 입력값 확인
    # --------------------------------------------------------

    validate_job_id(
        args.job_id
    )


    source_path = (
        resolve_path(
            args.source
        )
    )


    if not source_path.exists():

        raise FileNotFoundError(
            source_path
        )


    if (
        source_path.suffix.lower()
        != ".mp4"
    ):

        raise RuntimeError(
            "이번 실습에서는 "
            "MP4 파일만 사용합니다."
        )


    if not RUNTIME_SCRIPT.exists():

        raise FileNotFoundError(
            RUNTIME_SCRIPT
        )


    # --------------------------------------------------------
    # 2. 결과 폴더 준비
    # --------------------------------------------------------

    WEB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    JOB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUNTIME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    job_path = (
        JOB_DIR
        / args.job_id
    )


    job_path.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # 3. 기존 42번 Runtime 실행
    # --------------------------------------------------------

    command = [

        sys.executable,

        str(
            RUNTIME_SCRIPT
        ),

        "--source",
        str(
            source_path
        ),

        "--window-seconds",
        str(
            WINDOW_SECONDS
        ),

        "--confidence",
        str(
            CONFIDENCE_THRESHOLD
        ),

        "--stabilization-buffer",
        str(
            STABILIZATION_BUFFER
        ),

        "--output-dir",
        str(
            RUNTIME_DIR
        ),
    ]


    print(
        "42_realtime_action_video.py 실행"
    )


    completed = (
        subprocess.run(
            command,

            cwd=ROOT,

            text=True,

            capture_output=True,

            timeout=300,
        )
    )


    if completed.stdout:

        print(
            completed.stdout
        )


    if completed.returncode != 0:

        raise RuntimeError(
            "Runtime 실행에 "
            "실패했습니다.\n"
            + completed.stderr
        )


    # --------------------------------------------------------
    # 4. 42번이 만든 결과 파일명 확인
    # --------------------------------------------------------

    option_name = (

        f"w{WINDOW_SECONDS:g}_"

        f"c{CONFIDENCE_THRESHOLD:g}_"

        f"s{STABILIZATION_BUFFER}"
    )


    runtime_csv = (

        RUNTIME_DIR

        / (
            f"{source_path.stem}_"
            f"{option_name}.csv"
        )
    )


    runtime_video = (

        RUNTIME_DIR

        / (
            f"{source_path.stem}_"
            f"{option_name}.mp4"
        )
    )


    if not runtime_csv.exists():

        raise FileNotFoundError(
            runtime_csv
        )


    if not runtime_video.exists():

        raise FileNotFoundError(
            runtime_video
        )


    # --------------------------------------------------------
    # 5. Runtime CSV 읽기
    # --------------------------------------------------------

    df = pd.read_csv(
        runtime_csv
    )


    required_columns = {

        "video_time_sec",

        "raw_label",

        "threshold_label",

        "stable_label",

        "confidence",

        "prediction_updated",

        "processing_fps",
    }


    missing = (

        required_columns

        - set(
            df.columns
        )
    )


    if missing:

        raise RuntimeError(

            "Runtime CSV에 "
            "필요한 열이 없습니다: "

            + ", ".join(
                sorted(
                    missing
                )
            )
        )


    # --------------------------------------------------------
    # 6. 실제 Prediction이 만들어진 행만 선택
    # --------------------------------------------------------

    updates = (

        df.loc[
            prediction_mask(
                df[
                    "prediction_updated"
                ]
            )
        ]

        .copy()

        .reset_index(
            drop=True
        )
    )


    prediction_updates = int(
        len(
            updates
        )
    )


    # --------------------------------------------------------
    # 7. Prediction History 생성
    # --------------------------------------------------------

    history = []


    for _, row in updates.iterrows():

        history.append(
            {
                "video_time_sec":
                    round(
                        float(
                            row[
                                "video_time_sec"
                            ]
                        ),
                        3,
                    ),

                "raw_label":
                    str(
                        row[
                            "raw_label"
                        ]
                    ),

                "threshold_label":
                    str(
                        row[
                            "threshold_label"
                        ]
                    ),

                "stable_label":
                    str(
                        row[
                            "stable_label"
                        ]
                    ),

                "confidence":
                    round(
                        float(
                            row[
                                "confidence"
                            ]
                        ),
                        4,
                    ),
            }
        )


    # --------------------------------------------------------
    # 8. Final Prediction 계산
    # --------------------------------------------------------

    if updates.empty:

        detected_action = (
            "NO_PREDICTION"
        )

        latest_confidence = (
            None
        )

        latest_video_time = (
            None
        )

        final_match = (
            None
        )

        unknown_ratio = (
            None
        )

        stable_match_ratio = (
            None
        )

        mean_confidence = (
            None
        )


    else:

        last_row = (
            updates.iloc[-1]
        )


        detected_action = str(
            last_row[
                "stable_label"
            ]
        )


        latest_confidence = round(

            float(
                last_row[
                    "confidence"
                ]
            ),

            4,
        )


        latest_video_time = float(

            last_row[
                "video_time_sec"
            ]
        )


        final_match = (

            detected_action

            == args.expected
        )


        # ----------------------------------------------------
        # UNKNOWN 비율
        # ----------------------------------------------------

        threshold_labels = (

            updates[
                "threshold_label"
            ]

            .fillna("")

            .astype(str)
        )


        unknown_ratio = round(

            float(

                (
                    threshold_labels
                    == "UNKNOWN"
                ).mean()
            ),

            4,
        )


        # ----------------------------------------------------
        # Stable Label과 Expected Action 일치 비율
        # ----------------------------------------------------

        stable_labels = (

            updates[
                "stable_label"
            ]

            .fillna("")

            .astype(str)
        )


        stable_match_ratio = round(

            float(

                (
                    stable_labels
                    == args.expected
                ).mean()
            ),

            4,
        )


        # ----------------------------------------------------
        # 평균 Confidence
        # ----------------------------------------------------

        confidence_values = (

            pd.to_numeric(

                updates[
                    "confidence"
                ],

                errors="coerce",
            )

            .dropna()
        )


        if confidence_values.empty:

            mean_confidence = (
                None
            )

        else:

            mean_confidence = round(

                float(
                    confidence_values.mean()
                ),

                4,
            )


    # --------------------------------------------------------
    # 9. 평균 Processing FPS
    # --------------------------------------------------------

    mean_processing_fps = (

        positive_mean(
            df[
                "processing_fps"
            ]
        )
    )


    # --------------------------------------------------------
    # 10. Pose Skeleton 대표 Frame 생성
    # --------------------------------------------------------

    preview_path = (

        job_path

        / "preview.jpg"
    )


    write_pose_preview(

        source_path=(
            source_path
        ),

        preview_path=(
            preview_path
        ),

        video_time_sec=(
            latest_video_time
        ),
    )


    # --------------------------------------------------------
    # 11. Web에 전달할 결과 JSON
    # --------------------------------------------------------

    result = {

        "job_id":
            args.job_id,


        "created_at":
            (
                datetime.now()

                .astimezone()

                .isoformat(
                    timespec="seconds"
                )
            ),


        "source_file":
            source_path.name,


        "expected_action":
            args.expected,


        "expected_action_description":
            ACTION_INFO[
                args.expected
            ],


        "detected_action":
            detected_action,


        "detected_action_description":
            ACTION_INFO.get(
                detected_action,
                "",
            ),


        "latest_confidence":
            latest_confidence,


        "final_match":
            final_match,


        "metrics": {

            "prediction_updates":
                prediction_updates,

            "unknown_ratio":
                unknown_ratio,

            "stable_match_ratio":
                stable_match_ratio,

            "mean_confidence":
                mean_confidence,

            "mean_processing_fps":
                mean_processing_fps,
        },


        "runtime": {

            "window_seconds":
                WINDOW_SECONDS,

            "confidence_threshold":
                CONFIDENCE_THRESHOLD,

            "stabilization_buffer":
                STABILIZATION_BUFFER,
        },


        "runtime_csv":
            str(
                runtime_csv.relative_to(
                    ROOT
                )
            ),


        "processed_video":
            str(
                runtime_video.relative_to(
                    ROOT
                )
            ),


        "preview_image":
            str(
                preview_path.relative_to(
                    ROOT
                )
            ),
    }


    # --------------------------------------------------------
    # 12. job별 result.json 저장
    # --------------------------------------------------------

    result_path = (

        job_path

        / "result.json"
    )


    result_text = (

        json.dumps(

            result,

            ensure_ascii=False,

            indent=2,
        )
    )


    result_path.write_text(

        result_text,

        encoding="utf-8",
    )


    # --------------------------------------------------------
    # 13. 마지막 실행 결과도 별도로 저장
    # --------------------------------------------------------

    LATEST_PATH.write_text(

        result_text,

        encoding="utf-8",
    )


    HISTORY_PATH.write_text(

        json.dumps(

            history,

            ensure_ascii=False,

            indent=2,
        ),

        encoding="utf-8",
    )


    # --------------------------------------------------------
    # 14. 터미널 결과
    # --------------------------------------------------------

    print()

    print(
        "Expected Action    :",
        args.expected,
    )

    print(
        "Detected Action    :",
        detected_action,
    )

    print(
        "Latest Confidence  :",
        latest_confidence,
    )

    print(
        "Final Match        :",
        final_match,
    )

    print(
        "Prediction Updates :",
        prediction_updates,
    )

    print(
        "UNKNOWN Ratio      :",
        unknown_ratio,
    )

    print(
        "Stable Match Ratio :",
        stable_match_ratio,
    )

    print()

    print(
        "Saved:",
        result_path,
    )

    print(
        "Saved:",
        preview_path,
    )


if __name__ == "__main__":

    main()