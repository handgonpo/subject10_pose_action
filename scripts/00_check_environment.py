from __future__ import annotations

import platform
import sys
from pathlib import Path

import cv2
import torch
import ultralytics
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (
    ensure_day01_dirs,
    load_config,
    resolve_device,
)


def main() -> None:
    ensure_day01_dirs()

    config = load_config()
    device = resolve_device(config["device"])

    lines = [
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        f"OpenCV: {cv2.__version__}",
        f"PyTorch: {torch.__version__}",
        f"Ultralytics: {ultralytics.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"PyTorch CUDA: {torch.version.cuda}",
    ]

    if torch.cuda.is_available():
        lines.append(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
    else:
        lines.append("GPU: CPU mode")

    lines.append(f"Selected device: {device}")

    print("\n".join(lines))
    print("\nPose model loading...")

    YOLO(config["pose_model"])

    print("Pose model load: OK")

    report_path = (
        ROOT
        / "reports"
        / "day01"
        / "environment.txt"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()