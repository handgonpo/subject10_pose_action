from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "day03" / "images"

REQUIRED_IMAGES = [
    "arms_up_person_a.jpg",
    "arms_up_person_b.jpg",
    "arms_up_near.jpg",
    "arms_up_far.jpg",
    "left_arm_up.jpg",
    "right_arm_up.jpg",
    "sit_pose.jpg",
    "bend_pose.jpg",
]


def read_image(filename: str):
    path = IMAGE_DIR / filename

    if not path.exists():
        return None, f"[MISSING] {filename}"

    image = cv2.imread(str(path))

    if image is None:
        return None, f"[UNREADABLE] {filename}"

    height, width = image.shape[:2]

    return image, (
        f"[OK] {filename:22s} "
        f"{width} x {height}"
    )


def main() -> None:
    loaded = {}
    errors = []

    print(f"Image directory: {IMAGE_DIR}")
    print()

    for filename in REQUIRED_IMAGES:
        image, message = read_image(filename)
        print(message)

        if image is None:
            errors.append(filename)
        else:
            loaded[filename] = image

    if errors:
        raise RuntimeError(
            "3일차 입력 이미지의 파일명과 경로를 다시 확인하세요."
        )

    near = loaded["arms_up_near.jpg"]
    far = loaded["arms_up_far.jpg"]
    person_b = loaded["arms_up_person_b.jpg"]

    print()
    print("Comparison checks")

    same_canvas = near.shape[:2] == far.shape[:2]
    print(
        "near / far same canvas :",
        same_canvas,
    )

    same_person_b_size = (
        near.shape[:2]
        == person_b.shape[:2]
    )
    print(
        "person_b / near same size:",
        same_person_b_size,
    )

    same_pixels = False

    if near.shape == person_b.shape:
        same_pixels = np.array_equal(
            near,
            person_b,
        )

    print(
        "person_b / near identical :",
        same_pixels,
    )

    if not same_canvas:
        raise RuntimeError(
            "arms_up_near.jpg와 arms_up_far.jpg의 "
            "최종 이미지 해상도를 같게 맞추세요."
        )

    if not same_person_b_size:
        print(
            "[WARNING] person_b와 near의 "
            "이미지 크기가 다릅니다."
        )

    if not same_pixels:
        print(
            "[INFO] person_b와 near가 픽셀 단위로 "
            "완전히 같지는 않습니다. 같은 원본을 "
            "복사한 파일인지 확인하세요."
        )

    print()
    print("Day03 image data: READY")


if __name__ == "__main__":
    main()