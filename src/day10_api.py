from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import joblib

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from fastapi.responses import (
    FileResponse,
)


ROOT = Path(
    __file__
).resolve().parents[1]

INDEX_PATH = (
    ROOT
    / "web"
    / "day10"
    / "index.html"
)

MODEL_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_action_baseline.joblib"
)

FEATURE_PATH = (
    ROOT
    / "artifacts"
    / "day06"
    / "rf_feature_columns.json"
)

ANALYZE_SCRIPT = (
    ROOT
    / "scripts"
    / "67_build_web_result.py"
)

UPLOAD_DIR = (
    ROOT
    / "data"
    / "day10"
    / "web_uploads"
)

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

LATEST_PATH = (
    WEB_DIR
    / "latest_result.json"
)

HISTORY_PATH = (
    WEB_DIR
    / "prediction_history.json"
)


ACTION_INFO = {
    "bend_return": "허리를 굽혔다 다시 편다",
    "leg_raise_lower": "한쪽 다리를 들었다 내린다",
    "walk_turn_walk": "걷다가 돌아서 다시 걷는다",
    "drink_return": "음료수를 마시고 다시 앞을 본다",
    "sit_stand": "의자에 앉았다 다시 일어선다",
    "wave": "한 손을 들어 흔든다",
}


if not MODEL_PATH.exists():
    raise RuntimeError(
        f"모델 파일이 없습니다: {MODEL_PATH}"
    )

if not FEATURE_PATH.exists():
    raise RuntimeError(
        f"Feature 파일이 없습니다: {FEATURE_PATH}"
    )

if not ANALYZE_SCRIPT.exists():
    raise RuntimeError(
        f"67번 파일이 없습니다: {ANALYZE_SCRIPT}"
    )


MODEL = joblib.load(
    MODEL_PATH
)

FEATURE_COLUMNS = json.loads(
    FEATURE_PATH.read_text(
        encoding="utf-8"
    )
)


app = FastAPI(
    title=(
        "Day 10 Action Recognition"
    ),
    version="2.0.0",
)


UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

JOB_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def read_json(
    path: Path,
):
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"파일이 없습니다: "
                f"{path.name}"
            ),
        )

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=500,
            detail=(
                f"JSON 형식을 확인하세요: "
                f"{path.name}"
            ),
        ) from error


def checked_job_dir(
    job_id: str,
) -> Path:
    if not job_id.replace(
        "_",
        "",
    ).isalnum():
        raise HTTPException(
            status_code=400,
            detail=(
                "잘못된 job_id입니다."
            ),
        )

    return (
        JOB_DIR
        / job_id
    )


@app.get(
    "/",
    include_in_schema=False,
)
def dashboard():
    if not INDEX_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "web/day10/index.html이 없습니다."
            ),
        )

    return FileResponse(
        INDEX_PATH
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_type": (
            type(MODEL).__name__
        ),
        "feature_count": (
            len(FEATURE_COLUMNS)
        ),
        "supported_actions": (
            list(ACTION_INFO)
        ),
    }


@app.get("/api/actions")
def actions():
    return [
        {
            "label": label,
            "description": description,
        }
        for label, description
        in ACTION_INFO.items()
    ]


@app.get("/api/model")
def model_info():
    return {
        "model_file": (
            "artifacts/day06/"
            "rf_action_baseline.joblib"
        ),
        "model_type": (
            type(MODEL).__name__
        ),
        "feature_count": int(
            getattr(
                MODEL,
                "n_features_in_",
                -1,
            )
        ),
        "classes": [
            str(value)
            for value
            in MODEL.classes_
        ],
    }


@app.get("/api/latest")
def latest():
    return read_json(
        LATEST_PATH
    )


@app.get("/api/history")
def history():
    return read_json(
        HISTORY_PATH
    )


@app.get(
    "/api/result/{job_id}"
)
def result(
    job_id: str,
):
    path = (
        checked_job_dir(
            job_id
        )
        / "result.json"
    )

    return read_json(
        path
    )


@app.post("/api/analyze")
async def analyze(
    expected_action: str = Form(...),
    file: UploadFile = File(...),
):
    if expected_action not in ACTION_INFO:
        raise HTTPException(
            status_code=400,
            detail=(
                "지원하지 않는 Action입니다."
            ),
        )

    original_name = (
        file.filename
        or "upload.mp4"
    )

    suffix = (
        Path(
            original_name
        )
        .suffix
        .lower()
    )

    if suffix != ".mp4":
        raise HTTPException(
            status_code=400,
            detail=(
                "이번 실습에서는 "
                "MP4 파일만 업로드합니다."
            ),
        )

    job_id = (
        "job_"
        + uuid.uuid4().hex[:12]
    )

    upload_path = (
        UPLOAD_DIR
        / f"{job_id}.mp4"
    )

    try:
        with upload_path.open(
            "wb"
        ) as output:
            shutil.copyfileobj(
                file.file,
                output,
            )

    finally:
        await file.close()

    command = [
        sys.executable,
        str(ANALYZE_SCRIPT),
        "--source",
        str(upload_path),
        "--expected",
        expected_action,
        "--job-id",
        job_id,
    ]

    try:
        completed = (
            subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=300,
            )
        )

    except subprocess.TimeoutExpired as error:
        raise HTTPException(
            status_code=504,
            detail=(
                "영상 분석 시간이 "
                "5분을 초과했습니다."
            ),
        ) from error

    if completed.returncode != 0:
        detail = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or "영상 분석에 실패했습니다."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                detail[-2000:]
            ),
        )

    result_path = (
        checked_job_dir(
            job_id
        )
        / "result.json"
    )

    data = read_json(
        result_path
    )

    data[
        "original_filename"
    ] = original_name

    data[
        "preview_url"
    ] = (
        f"/media/preview/{job_id}"
    )

    return data


@app.get(
    "/media/preview/{job_id}",
    include_in_schema=False,
)
def preview(
    job_id: str,
):
    path = (
        checked_job_dir(
            job_id
        )
        / "preview.jpg"
    )

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Preview가 없습니다."
            ),
        )

    return FileResponse(
        path,
        media_type="image/jpeg",
    )