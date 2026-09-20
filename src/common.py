from __future__ import annotations

import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "project.json"


def load_config() -> dict:
    """project.json을 읽어 Python dict로 반환합니다."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_device(requested: str):
    """auto이면 CUDA 사용 가능 여부에 따라 GPU 또는 CPU를 선택합니다."""
    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return 0

    return "cpu"


def ensure_day01_dirs() -> None:
    """1일차에 필요한 폴더가 없으면 생성합니다."""
    directories = [
        PROJECT_ROOT / "data" / "day01" / "coco8_pose",
        PROJECT_ROOT / "data" / "day01" / "video",
        PROJECT_ROOT / "data" / "day01" / "ip_webcam",
        PROJECT_ROOT / "reports" / "day01",
        PROJECT_ROOT / "artifacts",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)