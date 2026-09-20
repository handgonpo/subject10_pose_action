from __future__ import annotations

import csv
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = (
    ROOT
    / "data"
    / "day04"
    / "videos"
)

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day04"
    / "tables"
    / "video_metadata.csv"
)

REQUIRED_VIDEOS = [
    "bend_return.mp4",
    "leg_raise_lower.mp4",
    "walk_turn_walk.mp4",
    "drink_return.mp4",
    "sit_stand.mp4",
    "wave.mp4",
]


def read_video_info(path: Path) -> dict:
    capture = cv2.VideoCapture(
        str(path)
    )

    if not capture.isOpened():
        return {
            "open_ok": 0,
            "read_ok": 0,
            "fps": 0.0,
            "frames": 0,
            "duration_sec": 0.0,
            "width": 0,
            "height": 0,
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

    success, first_frame = (
        capture.read()
    )

    read_ok = int(
        success
        and first_frame is not None
    )

    capture.release()

    duration = (
        frames / fps
        if fps > 0
        else 0.0
    )

    return {
        "open_ok": 1,
        "read_ok": read_ok,
        "fps": round(fps, 3),
        "frames": frames,
        "duration_sec": round(
            duration,
            3,
        ),
        "width": width,
        "height": height,
    }


def main() -> None:
    rows = []
    errors = []

    print(
        f"Video directory: "
        f"{VIDEO_DIR}"
    )

    print()

    for filename in REQUIRED_VIDEOS:
        path = VIDEO_DIR / filename

        if not path.exists():
            print(
                f"[MISSING] {filename}"
            )

            errors.append(filename)
            continue

        info = read_video_info(path)

        if not info["open_ok"]:
            print(
                f"[OPEN FAIL] "
                f"{filename}"
            )

            errors.append(filename)
            continue

        if not info["read_ok"]:
            print(
                f"[READ FAIL] "
                f"{filename}"
            )

            errors.append(filename)
            continue

        metadata_ok = (
            info["fps"] > 0
            and info["frames"] > 0
            and info["width"] > 0
            and info["height"] > 0
        )

        if not metadata_ok:
            print(
                f"[METADATA FAIL] "
                f"{filename}"
            )

            errors.append(filename)
            continue

        print(
            f"[OK] {filename:24s} "
            f"{info['width']}x"
            f"{info['height']} "
            f"fps={info['fps']:.3f} "
            f"frames={info['frames']} "
            f"duration="
            f"{info['duration_sec']:.3f}s"
        )

        rows.append(
            {
                "filename": filename,
                **info,
            }
        )

    if errors:
        raise RuntimeError(
            "4일차 입력 동영상을 "
            "다시 확인하세요: "
            + ", ".join(errors)
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
            fieldnames=[
                "filename",
                "open_ok",
                "read_ok",
                "fps",
                "frames",
                "duration_sec",
                "width",
                "height",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        "Day04 video data: READY"
    )


if __name__ == "__main__":
    main()