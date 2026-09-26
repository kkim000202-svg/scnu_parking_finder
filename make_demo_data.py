"""미니어처 주차장을 만들기 전에 앱을 바로 써볼 수 있도록 가상 주차장 사진을 만든다.

실행: python make_demo_data.py
결과: data/lots/demo/ 에 lot.json, empty.jpg, samples/*.jpg, labels.json, videos/demo.mp4 생성
"""

import json
import random

import cv2
import numpy as np

from occupancy import imwrite
from occupancy import Lot, Space, LOTS_DIR

W, H = 1200, 800
COLS, ROWS = 4, 2
SPACE_W, SPACE_H = 200, 280
LEFT, TOP, GAP_Y = 180, 60, 160
CAR_COLORS = [(40, 40, 200), (200, 200, 200), (30, 30, 30), (180, 90, 20), (60, 160, 230)]


def make_spaces():
    spaces = []
    for r in range(ROWS):
        for c in range(COLS):
            x1 = LEFT + c * SPACE_W
            y1 = TOP + r * (SPACE_H + GAP_Y)
            spaces.append(Space(str(len(spaces) + 1), x1 + 12, y1 + 12, x1 + SPACE_W - 12, y1 + SPACE_H - 12))
    return spaces


def draw_empty_lot(rng):
    img = np.full((H, W, 3), (95, 95, 95), np.uint8)
    noise = rng.integers(-12, 12, (H, W, 1), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for r in range(ROWS):
        y1 = TOP + r * (SPACE_H + GAP_Y)
        cv2.line(img, (LEFT, y1 + (SPACE_H if r == 0 else 0)),
                 (LEFT + COLS * SPACE_W, y1 + (SPACE_H if r == 0 else 0)), (240, 240, 240), 6)
        for c in range(COLS + 1):
            x = LEFT + c * SPACE_W
            cv2.line(img, (x, y1), (x, y1 + SPACE_H), (240, 240, 240), 6)
    return img


def draw_car_at(img, cx, cy, color, w=130, h=220, horizontal=False):
    """위에서 내려다본 차. horizontal=True면 옆으로 누운 방향(통로를 달리는 중)."""
    cx, cy = int(cx), int(cy)
    if horizontal:
        w, h = h, w
    x1, y1, x2, y2 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
    cv2.rectangle(img, (x1 + 4, y1 + 6), (x2 + 4, y2 + 6), (50, 50, 50), -1)  # 그림자
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    if horizontal:
        cv2.rectangle(img, (x2 - 85, y1 + 12), (x2 - 40, y2 - 12), (70, 50, 40), -1)  # 앞유리
        cv2.rectangle(img, (x1 + 25, y1 + 14), (x1 + 60, y2 - 14), (70, 50, 40), -1)  # 뒷유리
    else:
        cv2.rectangle(img, (x1 + 12, y1 + 40), (x2 - 12, y1 + 85), (70, 50, 40), -1)  # 앞유리
        cv2.rectangle(img, (x1 + 14, y2 - 60), (x2 - 14, y2 - 25), (70, 50, 40), -1)  # 뒷유리


def draw_car(img, space, color, rng):
    cx, cy = space.center
    draw_car_at(img, cx + int(rng.integers(-10, 10)), cy, color,
                130 + int(rng.integers(-8, 8)), 220 + int(rng.integers(-10, 10)))


def make_demo_video(path, empty, spaces, fps=8):
    """차 한 대가 통로를 따라 들어와 3번 칸에 주차하고, 6번 칸 차가 빠져나가는 17초 영상."""
    by_id = {s.id: s for s in spaces}
    aisle_y = TOP + SPACE_H + GAP_Y // 2
    parked = {"1": CAR_COLORS[0], "2": CAR_COLORS[1], "5": CAR_COLORS[2], "6": CAR_COLORS[3], "7": CAR_COLORS[4]}
    arriving, leaving = (60, 200, 90), CAR_COLORS[3]

    def lerp(a, b, t):
        return a + (b - a) * min(max(t, 0), 1)

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(17 * fps):
        t = i / fps
        frame = empty.copy()
        cars = dict(parked)
        s3x, s3y = by_id["3"].center
        s6x, s6y = by_id["6"].center
        if t >= 7:
            cars["3"] = arriving
        if t >= 10:
            cars.pop("6")
        for sid, color in cars.items():
            cx, cy = by_id[sid].center
            draw_car_at(frame, cx, cy, color)
        if 3 <= t < 5.5:    # 입구에서 통로를 따라 3번 칸 앞까지
            draw_car_at(frame, lerp(-120, s3x, (t - 3) / 2.5), aisle_y, arriving, horizontal=True)
        elif 5.5 <= t < 7:  # 3번 칸으로 들어가기
            draw_car_at(frame, s3x, lerp(aisle_y, s3y, (t - 5.5) / 1.5), arriving)
        if 10 <= t < 11.5:  # 6번 칸에서 통로로 나오기
            draw_car_at(frame, s6x, lerp(s6y, aisle_y, (t - 10) / 1.5), leaving)
        elif 11.5 <= t < 13:  # 통로를 따라 입구로 나가기
            draw_car_at(frame, lerp(s6x, -150, (t - 11.5) / 1.5), aisle_y, leaving, horizontal=True)
        writer.write(frame)
    writer.release()


def main():
    rng = np.random.default_rng(7)
    random.seed(7)
    folder = LOTS_DIR / "demo"
    (folder / "samples").mkdir(parents=True, exist_ok=True)

    spaces = make_spaces()
    # 두 줄 주차 칸 사이의 가로 통로 + 오른쪽 끝 세로 통로
    aisle_top = TOP + SPACE_H + 12
    aisles = [
        Space("a1", 0, aisle_top, W, aisle_top + GAP_Y - 24),
        Space("a2", LEFT + COLS * SPACE_W + 12, 0, W, H),
    ]
    lot = Lot(id="demo", name="데모 주차장 (순천대학교)", lat=34.9690, lng=127.4798,
              image_width=W, image_height=H, entrance=[60, H // 2], spaces=spaces, aisles=aisles)
    lot.save()

    empty = draw_empty_lot(rng)
    imwrite(folder / "empty.jpg", empty)

    labels = {}
    # (주차된 칸, 밝기 배율, 밝기 더하기, 색 틀어짐(B, G, R 배율))
    scenarios = {
        "sample_1.jpg": (["1", "2", "5", "6", "7"], 1.0, 0, (1, 1, 1)),
        "sample_2.jpg": (["1", "2", "3", "4", "5", "6", "7", "8"], 1.0, 0, (1, 1, 1)),
        "sample_3.jpg": (["3", "8"], 1.0, 0, (1, 1, 1)),
        "sample_4.jpg": (["1", "2", "3", "4", "5", "6", "8"], 1.0, 0, (1, 1, 1)),
        # 폰 카메라 자동 노출로 화면 전체가 밝아지거나 어두워진 경우
        "sample_5_bright.jpg": (["2", "7"], 1.35, 20, (1, 1, 1)),
        "sample_6_dark.jpg": (["1", "4", "5", "8"], 0.65, -5, (1, 1, 1)),
        # 형광등/전구색 조명으로 색이 틀어진 경우
        "sample_7_warm.jpg": (["3", "6"], 1.0, 0, (0.8, 1.0, 1.25)),
    }
    for name, (parked, alpha, beta, cast) in scenarios.items():
        img = cv2.convertScaleAbs(empty, alpha=1.0, beta=int(rng.integers(-8, 8)))
        for s in spaces:
            if s.id in parked:
                draw_car(img, s, random.choice(CAR_COLORS), rng)
        img = np.clip(img.astype(np.float32) * np.array(cast, np.float32) * alpha + beta, 0, 255)
        imwrite(folder / "samples" / name, img.astype(np.uint8))
        labels[name] = {s.id: s.id in parked for s in spaces}

    (folder / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    (folder / "videos").mkdir(exist_ok=True)
    make_demo_video(folder / "videos" / "demo.mp4", empty, spaces)
    print(f"데모 데이터 생성 완료: {folder}")


if __name__ == "__main__":
    main()
