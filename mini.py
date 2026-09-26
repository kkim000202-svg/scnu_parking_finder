"""책상 위 미니 주차장 (발표 시연용 실시간 데모).

좌표 기준: 네 모서리 마커(0 왼쪽 위, 1 오른쪽 위, 2 오른쪽 아래, 3 왼쪽 아래)의 '가운데'를 꼭짓점으로 하는 사각형.
카메라가 비스듬해도 마커 4개로 사진을 위에서 본 모습(이 사각형)으로 편다. 마커 4번은 입구(선택).

두 가지로 쓸 수 있다.
1) 인쇄용 도안(mini_template_A3/A4.pdf): 칸 위치가 정해져 있어 바로 동작.
2) 직접 그린·만든 주차장: 마커 종이(mini_markers_A4.pdf)를 오려 네 모서리에 붙이고,
   빈 주차장을 비춘 채 '칸 자동 인식'을 누르면 그려진 네모 칸을 찾아 저장한다 (calibrate).

실행: python mini.py → 도안 PDF, 마커 PDF, data/lots/mini_lot/lot.json(인쇄 도안용)
"""

import json
from pathlib import Path

import cv2
import numpy as np

from occupancy import LOTS_DIR, Space, imwrite, plan_routes, recommend

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
CORNER_IDS = (0, 1, 2, 3)          # 왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래
ENTRANCE_ID = 4
CANVAS_LONG = 1600                 # 펴진 사진의 긴 변 (px)
LOT_ID = "mini_lot"
LOT_NAME = "순천대 70주년기념관 뒤 주차장 (실시간 미니어처)"
LATLNG = (34.96994, 127.48299)

OCCUPIED_RATIO = 0.12   # 칸 안쪽에서 종이 색이 아닌 부분이 이 비율을 넘으면 주차됨
COLOR_DIFF = 28         # 종이 색과 이만큼(Lab 거리) 다르면 '종이가 아님'

# ---------- 인쇄용 도안 (mm, A3 가로) ----------
PAGE_MM = (420, 297)
MARKER_MM = 32
MARKER_MARGIN_MM = 8
SPACE_W, SPACE_D = 45, 90           # 칸 한 개 (핫휠 같은 1:64 장난감 차 기준)
COLS = 6
ROW_TOP_Y = 28
AISLE_H = 60
BLOCK_X = (PAGE_MM[0] - SPACE_W * COLS) / 2


def _template_marker_origins_mm():
    m, s = MARKER_MARGIN_MM, MARKER_MM
    W, H = PAGE_MM
    return [(m, m), (W - m - s, m), (W - m - s, H - m - s), (m, H - m - s)]


def layout_mm():
    """인쇄 도안의 칸·입구 위치 (mm)."""
    spaces = []
    aisle_y = ROW_TOP_Y + SPACE_D
    bottom_y = aisle_y + AISLE_H
    for row, y in enumerate((ROW_TOP_Y, bottom_y)):
        for c in range(COLS):
            x = BLOCK_X + c * SPACE_W
            spaces.append((str(row * COLS + c + 1), x, y, x + SPACE_W, y + SPACE_D))
    entrance = (MARKER_MARGIN_MM + MARKER_MM / 2, aisle_y + AISLE_H / 2)
    return spaces, entrance


def canvas_size(width, height):
    k = CANVAS_LONG / max(width, height)
    return int(round(width * k)), int(round(height * k))


def save_lot(spaces, entrance, size, source):
    """미니 주차장 설정 저장. spaces: [(id, [[x, y], ...]), ...] (펴진 사진 좌표). 지도에서는 70주년관 뒤 주차장 자리."""
    W, H = size
    lot = {
        "id": LOT_ID, "name": LOT_NAME, "lat": LATLNG[0], "lng": LATLNG[1],
        "image_width": W, "image_height": H,
        "entrance": [int(round(entrance[0])), int(round(entrance[1]))],
        "spaces": [Space.from_points(sid, pts).__dict__ for sid, pts in spaces],
        "aisles": [{"id": "a1", "x1": 0, "y1": 0, "x2": W, "y2": H, "points": None}],  # 칸 밖은 모두 길
        "auto_align": False, "entrance_latlng": list(LATLNG), "live": source,
    }
    folder = LOTS_DIR / LOT_ID
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "lot.json").write_text(json.dumps(lot, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder


def write_template_lot():
    """인쇄 도안 기준 설정 (마커 가운데 사이 사각형 좌표로 변환)."""
    origins = _template_marker_origins_mm()
    x0, y0 = origins[0][0] + MARKER_MM / 2, origins[0][1] + MARKER_MM / 2
    wmm, hmm = PAGE_MM[0] - 2 * x0, PAGE_MM[1] - 2 * y0
    W, H = canvas_size(wmm, hmm)
    k = W / wmm
    spaces, entrance = layout_mm()
    conv = lambda x, y: [(x - x0) * k, (y - y0) * k]  # noqa: E731
    polys = [(sid, [conv(x1, y1), conv(x2, y1), conv(x2, y2), conv(x1, y2)]) for sid, x1, y1, x2, y2 in spaces]
    return save_lot(polys, conv(*entrance), (W, H), "template")


# ---------- 인쇄물 ----------

def _font(size):
    from PIL import ImageFont
    for path in ("/System/Library/Fonts/AppleSDGothicNeo.ttc", "C:/Windows/Fonts/malgun.ttf",
                 "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _paste_marker(img, mid, x, y, size):
    from PIL import Image
    marker = cv2.aruco.generateImageMarker(DICT, mid, size, borderBits=1)
    img.paste(Image.fromarray(marker).convert("RGB"), (x, y))


def render_template(dpi=300):
    """인쇄용 도안 (A3). 실제 크기로 인쇄해야 칸이 장난감 차에 맞는다."""
    from PIL import Image, ImageDraw

    k = dpi / 25.4
    img = Image.new("RGB", (round(PAGE_MM[0] * k), round(PAGE_MM[1] * k)), "white")
    d = ImageDraw.Draw(img)
    font, small = _font(round(7 * k)), _font(round(4.2 * k))
    spaces, entrance = layout_mm()
    aisle_top = ROW_TOP_Y + SPACE_D
    for sid, x1, y1, x2, y2 in spaces:
        d.rectangle([x1 * k, y1 * k, x2 * k, y2 * k], outline=(40, 40, 40), width=max(2, round(1.2 * k)))
        ty = (y2 + 3) if y1 < aisle_top else (y1 - 10)  # 칸 번호는 칸 바깥 통로 쪽에
        d.text(((x1 + x2) / 2 * k, ty * k), sid, fill=(120, 120, 120), font=font, anchor="ma")
    ex, ey = entrance
    d.text((ex * k, (ey - 4) * k), "입구", fill=(20, 20, 20), font=font, anchor="mm")
    d.text((ex * k, (ey + 6) * k), "▶", fill=(20, 20, 20), font=font, anchor="mm")
    d.text((PAGE_MM[0] / 2 * k, (PAGE_MM[1] - 6) * k),
           "AI 빈자리 주차 안내 · 미니 주차장 — 네 모서리 마커가 가려지지 않게 두세요 (A3 실제 크기 인쇄)",
           fill=(150, 150, 150), font=small, anchor="mm")
    for mid, (x, y) in zip(CORNER_IDS, _template_marker_origins_mm()):
        _paste_marker(img, mid, round(x * k), round(y * k), round(MARKER_MM * k))
    return img


def render_marker_sheet(dpi=300):
    """직접 만든 주차장용 마커 5장 (A4 세로). 오려서 네 모서리와 입구에 붙인다."""
    from PIL import Image, ImageDraw

    k = dpi / 25.4
    img = Image.new("RGB", (round(210 * k), round(297 * k)), "white")
    d = ImageDraw.Draw(img)
    font, small = _font(round(6 * k)), _font(round(4.2 * k))
    size = 40
    labels = ["① 왼쪽 위", "② 오른쪽 위", "③ 오른쪽 아래", "④ 왼쪽 아래", "입구"]
    cells = [(30, 30), (125, 30), (125, 115), (30, 115), (77, 200)]
    for mid, label, (x, y) in zip((*CORNER_IDS, ENTRANCE_ID), labels, cells):
        # 오릴 선 (마커 둘레에 흰 여백을 남기고 자른다)
        d.rectangle([(x - 8) * k, (y - 8) * k, (x + size + 8) * k, (y + size + 16) * k], outline=(180, 180, 180), width=2)
        _paste_marker(img, mid, round(x * k), round(y * k), round(size * k))
        d.text(((x + size / 2) * k, (y + size + 4) * k), label, fill=(30, 30, 30), font=font, anchor="ma")
    d.text((105 * k, 12 * k), "미니 주차장 마커 — 회색 선을 따라 오려서 주차장 네 모서리(①~④)와 입구에 붙이세요",
           fill=(90, 90, 90), font=small, anchor="mm")
    d.text((105 * k, 270 * k), "마커 둘레의 흰 여백을 남겨 주세요 · 구기거나 접지 마세요 · 번호 위치가 바뀌면 안 돼요",
           fill=(90, 90, 90), font=small, anchor="mm")
    return img


# ---------- 실시간 판정 ----------

_detector = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())


def find_markers(frame):
    """{마커 번호: 가운데 좌표}"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).mean(axis=0) for i, c in zip(ids.flatten(), corners)}


def rectify(frame, size, markers=None):
    """마커 4개 가운데를 꼭짓점으로 사진을 편다. (펴진 사진, 변환 행렬, 찾은 마커) — 못 찾으면 (None, None, 마커)."""
    markers = find_markers(frame) if markers is None else markers
    if not all(i in markers for i in CORNER_IDS):
        return None, None, markers
    W, H = size
    src = np.float32([markers[i] for i in CORNER_IDS])
    dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, matrix, (W, H), borderValue=(40, 40, 40)), matrix, markers


def _shrink(points, scale):
    pts = np.asarray(points, np.float32)
    c = pts.mean(axis=0)
    return (c + (pts - c) * scale).astype(np.int32)


def judge(lot, top):
    """펴진 사진에서 칸마다 '종이가 아닌 것'이 덮은 비율로 빈칸 판정."""
    lab = cv2.cvtColor(cv2.GaussianBlur(top, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = top.shape[:2]
    statuses, ratios = {}, {}
    for s in lot.spaces:
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [_shrink(s.polygon, 0.7)], 1)  # 그려진 테두리 선은 빼고 칸 안쪽 70%만
        patch = lab[mask > 0]
        if not len(patch):
            continue
        # 종이 색은 칸 둘레에서 잰다 → 조명이 고르지 않아도 칸마다 맞춰짐 (밝은 쪽 = 종이)
        pad = int(min(s.x2 - s.x1, s.y2 - s.y1) * 0.35)
        ring = lab[max(0, s.y1 - pad):s.y2 + pad, max(0, s.x1 - pad):s.x2 + pad].reshape(-1, 3)
        paper = np.median(ring[ring[:, 0] >= np.percentile(ring[:, 0], 60)], axis=0)
        ratio = float((np.linalg.norm(patch - paper, axis=1) > COLOR_DIFF).mean())
        ratios[s.id] = ratio
        statuses[s.id] = ratio > OCCUPIED_RATIO
    return statuses, ratios


def analyze_live(lot, frame):
    """카메라 사진 한 장 → (분석 결과, 펴진 사진, 찾은 마커 수). 결과 형식은 occupancy.analyze와 같다."""
    top, matrix, markers = rectify(frame, (lot.image_width, lot.image_height))
    found = sum(1 for i in CORNER_IDS if i in markers)
    if top is None or not lot.spaces:
        return None, None, found
    entrance = tuple(lot.entrance)
    if ENTRANCE_ID in markers:  # 입구 마커가 있으면 그 위치를 입구로
        entrance = tuple(cv2.perspectiveTransform(np.float32([[markers[ENTRANCE_ID]]]), matrix)[0, 0])
    statuses, ratios = judge(lot, top)
    routes = plan_routes(lot.spaces, lot.aisles, entrance, (top.shape[1], top.shape[0]), exclude_spaces=True)
    return {
        "spaces": lot.spaces, "aisles": lot.aisles, "routes": routes, "entrance": entrance,
        "statuses": statuses, "recommended": recommend(lot.spaces, statuses, entrance, routes),
        "boxes": [], "ai_statuses": {}, "diff_ratios": ratios, "cls_probs": {},
        "empty_count": sum(1 for v in statuses.values() if not v), "total": len(lot.spaces),
        "camera_shift": 0.0,
    }, top, found


# ---------- 직접 그린 주차장: 칸 자동 인식 ----------

def detect_cells(top):
    """펴진 사진에서 선으로 그려진 닫힌 네모(칸)들을 찾는다. [[x, y] x4, ...]"""
    h, w = top.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(top, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    lines = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 12)
    lines = cv2.morphologyEx(lines, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))       # 점 같은 잡티 제거
    k = max(5, int(max(w, h) * 0.012))
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))  # 손으로 그린 선의 작은 틈 메우기
    count, labels, stats, _ = cv2.connectedComponentsWithStats(255 - lines, connectivity=4)
    cells = []
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue  # 가장자리에 닿은 영역 = 칸 바깥 종이
        if not (w * h * 0.004 < area < w * h * 0.3):
            continue
        contour, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour[0])
        if area / max(1, rw * rh) < 0.75 or min(rw, rh) / max(rw, rh) < 0.2:
            continue  # 네모가 아닌 모양은 뺀다
        box = cv2.boxPoints(((cx, cy), (rw + k, rh + k), angle))  # 선을 두껍게 만든 만큼 되돌린다
        cells.append(box.tolist())
    # 번호: 위쪽 줄부터, 왼쪽부터
    cells.sort(key=lambda b: np.mean(b, axis=0)[1])
    rows, row = [], []
    for b in cells:
        cy, hh = np.mean(b, axis=0)[1], np.ptp(np.array(b)[:, 1])
        if row and abs(cy - np.mean([np.mean(r, axis=0)[1] for r in row])) > hh * 0.5:
            rows.append(row)
            row = []
        row.append(b)
    if row:
        rows.append(row)
    return [b for r in rows for b in sorted(r, key=lambda b: np.mean(b, axis=0)[0])]


def calibrate(frame):
    """빈 미니 주차장 사진으로 칸을 자동 인식해 저장. (칸 개수, 펴진 사진, 안내 문구)"""
    markers = find_markers(frame)
    missing = [str(i + 1) for i in CORNER_IDS if i not in markers]
    if missing:
        return 0, None, f"모서리 마커 {', '.join(missing)}번이 안 보여요. 네 모서리 마커가 모두 화면에 보이게 해 주세요."
    c = [markers[i] for i in CORNER_IDS]
    width = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2
    height = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2
    size = canvas_size(width, height)
    top, matrix, _ = rectify(frame, size, markers)
    cells = detect_cells(top)
    if not cells:
        return 0, top, "칸을 못 찾았어요. 칸 테두리 선이 끊기지 않게 진하게 그려 주세요."
    if ENTRANCE_ID in markers:
        entrance = cv2.perspectiveTransform(np.float32([[markers[ENTRANCE_ID]]]), matrix)[0, 0].tolist()
    else:
        entrance = [size[0] * 0.02, size[1] / 2]  # 입구 마커가 없으면 왼쪽 가운데
    folder = save_lot([(str(i + 1), b) for i, b in enumerate(cells)], entrance, size, "custom")
    imwrite(folder / "empty.jpg", top)
    return len(cells), top, f"칸 {len(cells)}개를 찾았어요."


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    write_template_lot()
    img = render_template()
    img.save(root / "mini_template_A3.pdf", resolution=300)
    img.save(root / "mini_template_A4.pdf", resolution=300 * PAGE_MM[0] / 297)  # A4에 꽉 차게 (70% 축소)
    render_marker_sheet().save(root / "mini_markers_A4.pdf", resolution=300)
    print("도안:", root / "mini_template_A3.pdf", "/ A4 / 마커:", root / "mini_markers_A4.pdf")
