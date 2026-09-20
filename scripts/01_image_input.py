from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help="읽을 이미지 파일 경로",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.source)

    if not image_path.exists():
        raise FileNotFoundError(
            f"파일이 없습니다: {image_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"OpenCV가 이미지를 읽지 못했습니다: {image_path}"
        )

    height, width = image.shape[:2]

    print(f"Image path : {image_path}")
    print(f"Width      : {width}")
    print(f"Height     : {height}")
    print(f"Channels   : {image.shape[2]}")
    print(f"Shape      : {image.shape}")
    print(f"DataType   : {image.dtype}")


if __name__ == "__main__":
    main()