from __future__ import annotations

import csv
import re
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]

VIDEO_ROOT = ROOT / "data" / "day05" / "videos"
OUTPUT_PATH = (
    ROOT
    / "reports"
    / "day05"
    / "tables"
    / "video_inventory.csv"
)

SUBJECTS = ["S01", "S02", "S03"]
ACTIONS = [
    "A001",
    "A002",
    "A003",
    "A004",
    "A005",
    "A006",
]
TAKE = "T01"

FILENAME_PATTERN = re.compile(
    r"^(S\d{2})_(A\d{3})_(T\d{2})\.mp4$",
    re.IGNORECASE,
)


def expected_paths() -> set[Path]:
    paths = set()

    for subject in SUBJECTS:
        for action in ACTIONS:
            filename = (
                f"{subject}_"
                f"{action}_"
                f"{TAKE}.mp4"
            )
            paths.add(
                VIDEO_ROOT
                / subject
                / filename
            )

    return paths


def read_video_info(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        return {
            "open_ok": 0,
            "read_ok": 0,
            "fps": 0.0,
            "frames": 0,
            "width": 0,
            "height": 0,
            "duration_sec": 0.0,
        }

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
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

    success, first_frame = capture.read()
    capture.release()

    read_ok = int(
        success
        and first_frame is not None
    )

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
        "width": width,
        "height": height,
        "duration_sec": round(
            duration,
            3,
        ),
    }


def main() -> None:
    expected = expected_paths()
    actual = set(
        VIDEO_ROOT.rglob("*.mp4")
    )

    missing = sorted(
        expected - actual
    )
    unexpected = sorted(
        actual - expected
    )

    if missing:
        print("[MISSING]")
        for path in missing:
            print(path.relative_to(ROOT))

    if unexpected:
        print("[UNEXPECTED]")
        for path in unexpected:
            print(path.relative_to(ROOT))

    if missing or unexpected:
        raise RuntimeError(
            "5일차 MP4 파일명과 "
            "폴더 구조를 다시 확인하세요."
        )

    rows = []
    problems = []

    for path in sorted(actual):
        match = FILENAME_PATTERN.match(
            path.name
        )

        if match is None:
            problems.append(
                f"파일명 규칙 오류: "
                f"{path.name}"
            )
            continue

        subject = (
            match.group(1).upper()
        )
        action_id = (
            match.group(2).upper()
        )
        take = (
            match.group(3).upper()
        )

        if path.parent.name != subject:
            problems.append(
                "폴더 Subject와 "
                "파일명 Subject가 다름: "
                f"{path}"
            )

        info = read_video_info(path)

        metadata_ok = (
            info["open_ok"] == 1
            and info["read_ok"] == 1
            and info["fps"] > 0
            and info["frames"] > 0
            and info["width"] > 0
            and info["height"] > 0
        )

        if not metadata_ok:
            problems.append(
                "영상 읽기 또는 "
                "Metadata 오류: "
                f"{path.name}"
            )

        print(
            f"[OK] {path.name:20s} "
            f"{info['width']}x"
            f"{info['height']} "
            f"fps={info['fps']:.3f} "
            f"frames={info['frames']}"
        )

        rows.append(
            {
                "subject": subject,
                "action_id": action_id,
                "take": take,
                "video_path": str(
                    path.relative_to(ROOT)
                ),
                **info,
            }
        )

    if problems:
        for problem in problems:
            print("[ERROR]", problem)

        raise RuntimeError(
            "5일차 입력 영상을 "
            "다시 확인하세요."
        )

    if len(rows) != 18:
        raise RuntimeError(
            "5일차 기본 데이터는 "
            f"18개여야 합니다. "
            f"현재 {len(rows)}개입니다."
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
        writer.writerows(rows)

    print()
    print(f"Total clips: {len(rows)}")
    print(f"Saved: {OUTPUT_PATH}")
    print("Day05 video data: READY")


if __name__ == "__main__":
    main()