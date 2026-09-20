from __future__ import annotations

from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "day02" / "images"

REQUIRED_IMAGES = [
    "front.jpg",
    "arms_up.jpg",
    "side.jpg",
    "partial_body.jpg",
    "occlusion.jpg",
    "distance_far.jpg",
]


def main() -> None:
    missing = []
    unreadable = []

    print(f"Image directory: {IMAGE_DIR}")
    print()

    for filename in REQUIRED_IMAGES:
        image_path = IMAGE_DIR / filename

        if not image_path.exists():
            missing.append(filename)
            print(f"[MISSING] {filename}")
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            unreadable.append(filename)
            print(f"[UNREADABLE] {filename}")
            continue

        height, width = image.shape[:2]

        print(
            f"[OK] {filename:18s} "
            f"{width} x {height}"
        )

    print()

    if missing or unreadable:
        if missing:
            print("Missing:", ", ".join(missing))

        if unreadable:
            print("Unreadable:", ", ".join(unreadable))

        raise RuntimeError(
            "2일차 입력 이미지를 다시 확인하세요."
        )

    print("Day02 image data: READY")


if __name__ == "__main__":
    main()