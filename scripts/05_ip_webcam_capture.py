from __future__ import annotations

import argparse
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = ROOT / "data" / "day01" / "ip_webcam"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        required=True,
        help="IP Webcam video URL",
    )

    parser.add_argument(
        "--one-frame",
        action="store_true",
        help="첫 Frame 한 장만 저장하고 종료",
    )

    return parser.parse_args()


def save_image(
    frame,
    filename: str,
) -> Path:
    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = SAVE_DIR / filename

    saved = cv2.imwrite(
        str(output_path),
        frame,
    )

    if not saved:
        raise RuntimeError(
            f"이미지를 저장하지 못했습니다: {output_path}"
        )

    print(f"Saved: {output_path}")

    return output_path


def main() -> None:
    args = parse_args()

    capture = cv2.VideoCapture(args.url)

    if not capture.isOpened():
        raise RuntimeError(
            f"IP Webcam에 연결할 수 없습니다: {args.url}"
        )

    capture.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    success, frame = capture.read()

    if not success:
        capture.release()
        raise RuntimeError(
            "IP Webcam의 첫 Frame을 읽지 못했습니다."
        )

    height, width = frame.shape[:2]

    print(
        f"Connected: {width} x {height}"
    )

    if args.one_frame:
        save_image(
            frame,
            "connection_test.jpg",
        )

        capture.release()
        return

    print("1 : 정면 자세 저장")
    print("2 : 양팔 들기 저장")
    print("3 : 먼 거리 자세 저장")
    print("Q : 종료")

    while True:
        display_frame = frame.copy()

        cv2.putText(
            display_frame,
            "1: Front  2: Arms Up  3: Far  Q: Quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "Subject10 IP Webcam",
            display_frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("1"):
            save_image(
                frame,
                "front.jpg",
            )

        if key == ord("2"):
            save_image(
                frame,
                "arms_up.jpg",
            )

        if key == ord("3"):
            save_image(
                frame,
                "far.jpg",
            )

        success, frame = capture.read()

        if not success:
            print(
                "다음 Frame을 읽지 못했습니다."
            )
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()