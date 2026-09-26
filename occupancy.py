"""주차 칸 빈자리 판정, 추천, 결과 그리기.

주차장 설정은 data/lots/<주차장ID>/lot.json 에 저장된다.
칸 좌표는 기준 사진(empty.jpg) 크기 기준 픽셀 좌표이고,
다른 크기의 사진이 들어오면 자동으로 비율을 맞춘다.
"""

import heapq
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

LOTS_DIR = Path(__file__).parent / "data" / "lots"

# ---------- 파일 읽기·쓰기 (Windows 한글 경로 대응) ----------
# cv2.imread / cv2.imwrite / cv2.VideoCapture는 Windows에서 경로에 한글이 있으면 파일을 못 연다.
# 그래서 파일은 파이썬으로 읽고 쓰고, OpenCV에는 메모리에 올린 데이터만 넘긴다.

def imread(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(path, img, ext=None):
    ext = ext or Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
    return ok


def open_video(path):
    """영상 파일 열기. 한글 경로라 못 열면 임시 폴더(영어 경로)에 복사해서 연다."""
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened() or str(path).isascii():
        return cap
    import hashlib, shutil, tempfile
    tmp = Path(tempfile.gettempdir()) / f"parking_video_{hashlib.md5(str(path).encode()).hexdigest()[:10]}{Path(path).suffix}"
    if not tmp.exists() or tmp.stat().st_size != Path(path).stat().st_size:
        shutil.copyfile(path, tmp)
    return cv2.VideoCapture(str(tmp))


# 판정 방식
MODE_AI = "ai"          # YOLO가 찾은 차와 칸이 겹치는지
MODE_DIFF = "diff"      # 빈 주차장 기준 사진과 비교해 칸이 달라졌는지
MODE_BOTH = "both"      # 둘 중 하나라도 주차됨이면 주차됨
MODE_CLS = "cls"        # 칸 이미지를 보고 직접 판단하는, 실제 CCTV 사진으로 학습시킨 AI


@dataclass
class Space:
    """주차 칸(또는 통로). x1~y2는 감싸는 사각형이고,
    카메라가 비스듬해서 칸이 기울어져 보이면 points에 꼭짓점들을 저장한다."""
    id: str
    x1: int
    y1: int
    x2: int
    y2: int
    points: list | None = None

    @classmethod
    def from_points(cls, id, points):
        pts = np.asarray(points, float).reshape(-1, 2)
        if len(pts) < 3:
            raise ValueError(f"칸 {id}: 꼭짓점이 3개 이상 필요해요 (받은 개수 {len(pts)})")
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        return cls(id, int(x1), int(y1), int(math.ceil(x2)), int(math.ceil(y2)),
                   [[int(round(x)), int(round(y))] for x, y in pts])

    @property
    def polygon(self):
        if self.points:
            return np.array(self.points, np.int32)
        return np.array([[self.x1, self.y1], [self.x2, self.y1], [self.x2, self.y2], [self.x1, self.y2]], np.int32)

    @property
    def center(self):
        if self.points:
            m = cv2.moments(self.polygon)
            if m["m00"]:
                return (m["m10"] / m["m00"], m["m01"] / m["m00"])
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def contains(self, x, y):
        return cv2.pointPolygonTest(self.polygon.astype(np.float32), (float(x), float(y)), False) >= 0

    def region(self, shape):
        """사진 안으로 잘라낸 감싸는 사각형 (y1, y2, x1, x2)과 그 안에서 칸에 해당하는 마스크."""
        h, w = shape[:2]
        x1, y1 = max(0, self.x1), max(0, self.y1)
        x2, y2 = min(w, self.x2 + 1), min(h, self.y2 + 1)
        mask = np.zeros((max(0, y2 - y1), max(0, x2 - x1)), np.uint8)
        if mask.size:
            cv2.fillPoly(mask, [self.polygon - [x1, y1]], 1)
        return (y1, y2, x1, x2), mask.astype(bool)


@dataclass
class Lot:
    id: str
    name: str
    lat: float
    lng: float
    image_width: int
    image_height: int
    entrance: list = field(default_factory=lambda: [0, 0])
    spaces: list = field(default_factory=list)
    aisles: list = field(default_factory=list)  # 차가 다니는 통로 사각형들 (경로 계산용)
    # 기준 사진 비교 민감도: 픽셀 색 차이 기준(0~255), 칸 안에서 바뀐 면적 비율 기준(0~1)
    pixel_threshold: int = 40
    changed_ratio: float = 0.25
    auto_align: bool = True
    # 입구가 여러 개일 때: [{"id", "px": [x, y](사진 좌표), "latlng": [위도, 경도]}]. 차가 오는 방향에 가까운 입구를 쓴다
    entrances: list | None = None
    frames_from: str | None = None  # 다른 주차장의 CCTV 사진을 같이 씀 (카메라 하나가 주차장 두 곳을 비출 때)
    cls_threshold: float = 0.5  # 칸 분류 AI가 이 확률 이상이면 '주차됨' (CCTV마다 조정)
    live: str | None = None  # 'mini' = 카메라가 보내는 실시간 사진으로 판정 (mini.py)
    entrance_latlng: list | None = None  # 지도에서 주차장 입구 위치 [위도, 경도] (길안내 도착 지점)  # 카메라가 조금 움직여도 기준 사진과 맞춰 칸 위치를 자동 보정

    @property
    def folder(self):
        return LOTS_DIR / self.id

    @property
    def reference_path(self):
        return self.folder / "empty.jpg"

    def reference_image(self):
        if not self.reference_path.exists():
            return None
        return imread(self.reference_path)

    def save(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["spaces"] = [asdict(s) for s in self.spaces]
        data["aisles"] = [asdict(a) for a in self.aisles]
        (self.folder / "lot.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def list_lots():
    if not LOTS_DIR.exists():
        return []
    return sorted(p.name for p in LOTS_DIR.iterdir() if (p / "lot.json").exists())


def load_lot(lot_id):
    data = json.loads((LOTS_DIR / lot_id / "lot.json").read_text(encoding="utf-8"))
    data["spaces"] = [Space(**s) for s in data.get("spaces", [])]
    data["aisles"] = [Space(**a) for a in data.get("aisles", [])]
    return Lot(**data)


def scale_rects(rects, sx, sy):
    return [Space(r.id, int(r.x1 * sx), int(r.y1 * sy), int(r.x2 * sx), int(r.y2 * sy),
                  [[int(x * sx), int(y * sy)] for x, y in r.points] if r.points else None)
            for r in rects]


def scale_spaces(lot, frame):
    """기준 사진 크기로 저장된 칸·통로·입구 좌표를 현재 사진 크기에 맞춘다."""
    h, w = frame.shape[:2]
    sx = w / lot.image_width
    sy = h / lot.image_height
    entrance = (lot.entrance[0] * sx, lot.entrance[1] * sy)
    return scale_rects(lot.spaces, sx, sy), scale_rects(lot.aisles, sx, sy), entrance


# ---------- 통로를 따라가는 경로 ----------

_NEIGHBORS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def _shortest_paths(walkable, start):
    """격자에서 start부터 모든 칸까지의 최단 거리(다익스트라). 대각선 이동 시 모서리는 못 뚫는다."""
    gh, gw = walkable.shape
    dist = np.full((gh, gw), np.inf)
    prev = np.full((gh, gw, 2), -1, int)
    dist[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        d, (y, x) = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        for dy, dx, cost in _NEIGHBORS:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < gh and 0 <= nx < gw) or not walkable[ny, nx]:
                continue
            if dy and dx and not (walkable[y, nx] and walkable[ny, x]):
                continue
            nd = d + cost
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                prev[ny, nx] = (y, x)
                heapq.heappush(heap, (nd, (ny, nx)))
    return dist, prev


def plan_routes(spaces, aisles, entrance, image_size, grid_cells=80, exclude_spaces=False):
    """입구에서 통로를 따라 각 칸까지 가는 경로. exclude_spaces=True면 다른 칸 위로는 지나가지 않는다.

    반환: {칸 id: {"length": 이동 거리(px), "points": [(x, y), ...]}}
    통로가 없으면 빈 dict (그러면 직선 거리로 추천한다).
    """
    if not aisles:
        return {}
    w, h = image_size
    cell = max(w, h) / grid_cells
    gw, gh = math.ceil(w / cell), math.ceil(h / cell)
    xs = (np.arange(gw) + 0.5) * cell
    ys = (np.arange(gh) + 0.5) * cell
    grid_x, grid_y = np.meshgrid(xs, ys)

    walkable = np.zeros((gh, gw), bool)
    for a in aisles:
        inside = (grid_x >= a.x1) & (grid_x <= a.x2) & (grid_y >= a.y1) & (grid_y <= a.y2)
        if a.points:  # 기울어진 통로는 다각형 안쪽만
            ys_in, xs_in = np.nonzero(inside)
            for y, x in zip(ys_in, xs_in):
                inside[y, x] = a.contains(grid_x[y, x], grid_y[y, x])
        walkable |= inside
    if exclude_spaces:
        for s in spaces:
            inside = (grid_x >= s.x1) & (grid_x <= s.x2) & (grid_y >= s.y1) & (grid_y <= s.y2)
            for y, x in zip(*np.nonzero(inside)):
                if s.contains(grid_x[y, x], grid_y[y, x]):
                    walkable[y, x] = False
    if not walkable.any():
        return {}

    # 입구에서 가장 가까운 통로 격자에서 출발
    wy, wx = np.nonzero(walkable)
    nearest = np.argmin((xs[wx] - entrance[0]) ** 2 + (ys[wy] - entrance[1]) ** 2)
    start = (int(wy[nearest]), int(wx[nearest]))
    dist, prev = _shortest_paths(walkable, start)

    ry, rx = np.nonzero(np.isfinite(dist))  # 입구에서 갈 수 있는 격자들
    px, py = xs[rx], ys[ry]
    to_start = math.dist(entrance, (xs[start[1]], ys[start[0]]))

    routes = {}
    for s in spaces:
        # 칸에 맞닿은 통로 격자들 중 칸 가운데 정면에서 진입 (모서리로 비스듬히 들어가지 않도록)
        gap = np.hypot(np.maximum(np.maximum(s.x1 - px, px - s.x2), 0),
                       np.maximum(np.maximum(s.y1 - py, py - s.y2), 0))
        candidates = np.flatnonzero(gap <= gap.min() + cell)
        cx, cy = s.center
        goal_idx = candidates[np.argmin(np.hypot(px[candidates] - cx, py[candidates] - cy))]
        goal = (int(ry[goal_idx]), int(rx[goal_idx]))

        cells = [goal]
        while cells[-1] != start:
            cells.append(tuple(prev[cells[-1]]))
        cells.reverse()
        path = np.array([[xs[x], ys[y]] for y, x in cells], np.float32)
        if len(path) > 2:  # 계단 모양 격자 경로를 곧은 선분으로 정리
            path = cv2.approxPolyDP(path.reshape(-1, 1, 2), cell * 0.75, False).reshape(-1, 2)

        center = s.center
        points = [tuple(entrance)] + [tuple(map(float, pt)) for pt in path] + [center]
        length = to_start + dist[goal] * cell + math.dist(points[-2], center)
        routes[s.id] = {"length": float(length), "points": points}
    return routes


def _overlap_ratio(space, box):
    """칸 면적 중 차 박스와 겹치는 비율 (0~1)."""
    (y1, y2, x1, x2), mask = space.region((10**6, 10**6))
    if not mask.any():
        return 0.0
    bx1, by1 = int(max(box[0], x1)) - x1, int(max(box[1], y1)) - y1
    bx2, by2 = int(min(box[2], x2)) - x1, int(min(box[3], y2)) - y1
    if bx2 <= bx1 or by2 <= by1:
        return 0.0
    return float(mask[by1:by2, bx1:bx2].sum()) / float(mask.sum())


def judge_by_detection(spaces, boxes, min_overlap=0.3):
    """차 박스의 중심이 칸 안에 있거나, 칸을 충분히 덮으면 주차됨."""
    result = {}
    for s in spaces:
        occupied = False
        for b in boxes:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            inside = s.contains(cx, cy)
            if inside or _overlap_ratio(s, b) >= min_overlap:
                occupied = True
                break
        result[s.id] = occupied
    return result


def match_lighting(frame, reference, spaces):
    """폰 카메라의 자동 노출·조명 색 변화를 없애기 위해 frame의 밝기와 색을 reference에 맞춘다.

    차가 서 있지 않은 통로(칸 바깥) 부분의 색 평균을 기준으로 채널별 배율을 구한다.
    """
    mask = np.ones(frame.shape[:2], np.uint8)
    cv2.fillPoly(mask, [s.polygon for s in spaces], 0)
    mask = mask.astype(bool)
    if mask.mean() < 0.05:  # 칸이 화면을 거의 다 덮으면 전체 평균 사용
        mask[:] = True
    cur_mean = frame[mask].reshape(-1, 3).mean(axis=0)
    ref_mean = reference[mask].reshape(-1, 3).mean(axis=0)
    gain = np.clip(ref_mean / np.maximum(cur_mean, 1), 0.3, 3.0)
    return np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def judge_by_difference(spaces, frame, reference, pixel_threshold=40, changed_ratio=0.25):
    """빈 주차장 사진과 비교해 칸 안의 픽셀이 많이 바뀌었으면 주차됨.

    카메라 위치가 기준 사진을 찍을 때와 같아야 정확하다.
    """
    ref = cv2.resize(reference, (frame.shape[1], frame.shape[0]))
    frame = match_lighting(frame, ref, spaces)
    # 흑백으로만 비교하면 아스팔트와 밝기가 비슷한 차(예: 빨간 차)를 놓치므로 색 채널별로 비교
    cur = cv2.GaussianBlur(frame, (5, 5), 0)
    ref = cv2.GaussianBlur(ref, (5, 5), 0)
    diff = cv2.absdiff(cur, ref).max(axis=2)

    result, ratios = {}, {}
    for s in spaces:
        (y1, y2, x1, x2), mask = s.region(diff.shape)
        roi = diff[y1:y2, x1:x2][mask]
        ratio = float((roi > pixel_threshold).mean()) if roi.size else 0.0
        ratios[s.id] = ratio
        result[s.id] = ratio >= changed_ratio
    return result, ratios


def recommend(spaces, statuses, entrance, routes=None):
    """입구에서 가장 가까운 빈칸의 id. 통로 경로가 있으면 실제 이동 거리로 비교한다."""
    empty = [s for s in spaces if not statuses.get(s.id, False)]
    if not empty:
        return None

    def distance(s):
        if routes and s.id in routes:
            return routes[s.id]["length"]
        return math.dist(s.center, entrance)

    return min(empty, key=distance).id


# ---------- 카메라 흔들림 보정 ----------

_CLAHE = cv2.createCLAHE(2.0, (8, 8))


def estimate_camera_shift(frame, reference, work_width=960, model="affine", any_view=False):
    """기준 사진 좌표 → 현재 사진 좌표로 옮기는 3×3 변환 행렬.

    바닥 선·건물처럼 움직이지 않는 특징점(SIFT)을 두 사진에서 찾아 맞춘다.
    명암을 고르게 펴서(CLAHE) 날씨·시간대가 달라도 특징점이 맞도록 한다.
    카메라가 거의 안 움직였거나, 믿을 만한 변환을 못 찾으면 None.
    any_view=True: 크기·방향이 다른 사진(예: 세로로 찍은 폰 사진)도 허용하고 원근 변환(homography)으로 맞춘다.
    """
    h, w = frame.shape[:2]
    rh, rw = reference.shape[:2]
    scale = min(1.0, work_width / w)
    scale_ref = min(1.0, work_width / rw) if any_view else scale
    sift = cv2.SIFT_create(3000)
    gray = lambda img, sc: _CLAHE.apply(cv2.cvtColor(cv2.resize(img, None, fx=sc, fy=sc), cv2.COLOR_BGR2GRAY))
    kp_ref, des_ref = sift.detectAndCompute(gray(reference, scale_ref), None)
    kp_cur, des_cur = sift.detectAndCompute(gray(frame, scale), None)
    if any_view:
        model = "homography"
    if des_ref is None or des_cur is None or len(kp_ref) < 20 or len(kp_cur) < 20:
        return None

    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    pairs = matcher.knnMatch(des_ref, des_cur, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 15:
        return None
    src = np.float32([kp_ref[m.queryIdx].pt for m in good]) / scale_ref
    dst = np.float32([kp_cur[m.trainIdx].pt for m in good]) / scale
    if model == "homography":
        matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    else:  # 회전·이동·크기만: 매칭이 적어도 안정적
        affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        matrix = None if affine is None else np.vstack([affine, [0, 0, 1]])
    if matrix is None or inliers.sum() < 12 or inliers.mean() < 0.3:
        return None

    if any_view:
        return matrix
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    moved = np.linalg.norm(cv2.perspectiveTransform(corners, matrix) - corners, axis=2)
    if moved.max() < 3 or moved.max() > 0.3 * w:  # 거의 안 움직였거나, 말이 안 되게 크게 움직임
        return None
    return matrix


def _visible_ratio(space, shape):
    """칸 넓이 중 사진 안에 들어온 비율."""
    h, w = shape[:2]
    area = cv2.contourArea(space.polygon.astype(np.float32))
    if area <= 0:
        return 0.0
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [space.polygon], 1)
    return float(mask.sum() / area)


def warp_rects(rects, matrix):
    moved = []
    for r in rects:
        pts = cv2.perspectiveTransform(r.polygon.astype(np.float32).reshape(-1, 1, 2), matrix)
        moved.append(Space.from_points(r.id, pts.reshape(-1, 2)))
    return moved


def analyze(lot, frame, detector=None, mode=MODE_BOTH, conf=0.25, reference=None, classifier=None):
    """사진 한 장을 분석해 칸별 상태와 추천 칸을 돌려준다. reference를 안 주면 주차장의 기준 사진을 읽는다."""
    spaces, aisles, entrance = scale_spaces(lot, frame)
    h, w = frame.shape[:2]
    if reference is None and (lot.auto_align or mode in (MODE_DIFF, MODE_BOTH)):
        reference = lot.reference_image()

    shift = 0.0
    other_view = reference is not None and abs(w / h - reference.shape[1] / reference.shape[0]) > 0.05
    if lot.auto_align and other_view:
        # 기준 사진과 방향·비율이 다른 사진(세로로 찍은 폰 사진 등): 기준 사진 좌표계에서 바로 원근 변환을 찾는다
        matrix = estimate_camera_shift(frame, reference, any_view=True)
        if matrix is not None:
            rs, ra, re = scale_spaces(lot, reference)
            spaces, aisles = warp_rects(rs, matrix), warp_rects(ra, matrix)
            entrance = tuple(cv2.perspectiveTransform(np.float32([[re]]), matrix)[0, 0])
            reference = cv2.warpPerspective(reference, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
            shift = float(max(w, h))  # 다른 각도에서 찍은 사진을 통째로 맞춤
    elif reference is not None and reference.shape[:2] != (h, w):
        reference = cv2.resize(reference, (w, h))

    if lot.auto_align and reference is not None and not shift:
        matrix = estimate_camera_shift(frame, reference)
        if matrix is not None:
            spaces, aisles = warp_rects(spaces, matrix), warp_rects(aisles, matrix)
            entrance = tuple(cv2.perspectiveTransform(np.float32([[entrance]]), matrix)[0, 0])
            reference = cv2.warpPerspective(reference, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            shift = float(np.linalg.norm(cv2.perspectiveTransform(corners, matrix) - corners, axis=2).max())

    spaces = [sp for sp in spaces if _visible_ratio(sp, frame.shape) >= 0.6]  # 사진 밖으로 잘린 칸은 판정하지 않는다
    # 경로는 다른 주차 칸 위로 지나가지 않게 (통로를 넉넉히 그려도 칸은 피해 간다)
    routes = plan_routes(spaces, aisles, entrance, (frame.shape[1], frame.shape[0]), exclude_spaces=True)
    if spaces and aisles and not routes:  # 칸을 빼니 길이 끊기면 예전 방식으로
        routes = plan_routes(spaces, aisles, entrance, (frame.shape[1], frame.shape[0]))
    boxes = []
    statuses = {s.id: False for s in spaces}
    ai, ratios, cls_probs = {}, {}, {}

    if mode == MODE_CLS and classifier is not None:
        statuses, cls_probs = classifier.predict(frame, spaces, threshold=lot.cls_threshold)

    if mode in (MODE_AI, MODE_BOTH) and detector is not None:
        boxes = detector.detect(frame, conf=conf)
        ai = judge_by_detection(spaces, [b["box"] for b in boxes])
        statuses = {k: statuses[k] or v for k, v in ai.items()}

    if mode in (MODE_DIFF, MODE_BOTH) and reference is not None:
        diff, ratios = judge_by_difference(spaces, frame, reference,
                                           lot.pixel_threshold, lot.changed_ratio)
        statuses = {k: statuses[k] or v for k, v in diff.items()}

    best = recommend(spaces, statuses, entrance, routes)
    plates = find_plates(frame, spaces, statuses)
    return {
        "plates": plates,   # 번호판으로 보이는 곳 (화면에 보여줄 때 가린다)
        "spaces": spaces,
        "aisles": aisles,
        "routes": routes,
        "entrance": entrance,
        "statuses": statuses,
        "recommended": best,
        "boxes": boxes,
        "ai_statuses": ai,      # 칸별 AI 판정 (AI를 안 썼으면 빈 dict)
        "diff_ratios": ratios,  # 칸별 바뀐 면적 비율 (비교를 안 했으면 빈 dict)
        "cls_probs": cls_probs,  # 칸별 주차됨 확률 (칸 분류 AI를 안 썼으면 빈 dict)
        "empty_count": sum(1 for v in statuses.values() if not v),
        "total": len(spaces),
        "camera_shift": shift,  # 카메라가 움직여서 보정한 크기(px). 0이면 보정 안 함
    }


class StatusSmoother:
    """실시간 영상에서 손이나 그림자가 잠깐 지나갈 때 칸 상태가 깜빡이지 않게,
    같은 판정이 `frames`번 연속 나와야 상태를 바꾼다."""

    def __init__(self, frames=2):
        self.frames = frames
        self.stable = {}
        self.pending = {}  # 칸 id → (새 상태, 연속 횟수)

    def update(self, statuses):
        for sid, value in statuses.items():
            if sid not in self.stable or self.frames <= 1:
                self.stable[sid] = value
                self.pending.pop(sid, None)
            elif value == self.stable[sid]:
                self.pending.pop(sid, None)
            else:
                prev, count = self.pending.get(sid, (value, 0))
                count = count + 1 if prev == value else 1
                if count >= self.frames:
                    self.stable[sid] = value
                    self.pending.pop(sid, None)
                else:
                    self.pending[sid] = (value, count)
        for sid in set(self.stable) - set(statuses):  # 지워진 칸 정리
            self.stable.pop(sid)
        return {sid: self.stable[sid] for sid in statuses}


def apply_smoothing(result, smoother):
    """분석 결과의 칸 상태를 안정화하고 추천 칸과 빈칸 수를 다시 계산한다."""
    statuses = smoother.update(result["statuses"])
    return {
        **result,
        "statuses": statuses,
        "recommended": recommend(result["spaces"], statuses, result["entrance"],
                                 result.get("routes")),
        "empty_count": sum(1 for v in statuses.values() if not v),
    }


GREEN = (80, 200, 60)
RED = (60, 60, 230)
GOLD = (0, 200, 255)
BLUE = (230, 150, 40)


# ---------- 번호판 가리기 (개인정보) ----------

def find_plates(frame, spaces, statuses):
    """주차된 칸 안에서 번호판처럼 보이는 곳을 찾는다: 밝은 바탕에 진한 글자가 가로로 늘어선 납작한 네모.

    반환: [(x, y, w, h), ...]  (사진 픽셀 좌표). 정확한 인식이 아니라 '가릴 곳'을 넉넉히 찾는 용도.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gx = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3))  # 글자 = 가로 방향 밝기 변화가 촘촘함
    H, W = gray.shape
    found = []
    for sp in spaces:
        if not statuses.get(sp.id):
            continue
        x1, y1 = max(0, sp.x1), max(0, sp.y1)
        x2, y2 = min(W, sp.x2), min(H, sp.y2)
        sw, sh = x2 - x1, y2 - y1
        if sw < 40 or sh < 20:
            continue
        g = gx[y1:y2, x1:x2]
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(7, int(sw * 0.06)), 3)))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if h < 6 or w < sw * 0.1 or w > sw * 0.5 or not 1.8 <= w / h <= 6.5:
                continue
            if y < sh * 0.3:  # 번호판은 범퍼 쪽(칸 아래 2/3)에 있다
                continue
            patch = gray[y1 + y:y1 + y + h, x1 + x:x1 + x + w]
            if patch.mean() < 85 or patch.std() < 32:  # 밝은 바탕 + 진한 글자
                continue
            found.append((x1 + x, y1 + y, w, h))
    return found


def blur_plates(img, plates, pad=0.35):
    """번호판 영역을 알아볼 수 없게 뭉갠다 (테두리를 조금 넓혀서)."""
    out = img
    H, W = img.shape[:2]
    for x, y, w, h in plates:
        px, py = int(w * pad), int(h * pad) + 2
        ax, ay = max(0, x - px), max(0, y - py)
        bx, by = min(W, x + w + px), min(H, y + h + py)
        if bx - ax < 4 or by - ay < 4:
            continue
        roi = out[ay:by, ax:bx]
        small = cv2.resize(roi, (max(2, (bx - ax) // 10), max(2, (by - ay) // 5)), interpolation=cv2.INTER_AREA)
        out[ay:by, ax:bx] = cv2.GaussianBlur(cv2.resize(small, (bx - ax, by - ay), interpolation=cv2.INTER_LINEAR), (9, 9), 0)
    return out


def _edge_point(space, outside):
    """칸 중심에서 outside 점 방향으로 나갈 때 칸 테두리와 만나는 점 (이분 탐색)."""
    cx, cy = space.center
    if space.contains(*outside):
        return tuple(outside)
    lo, hi = 0.0, 1.0
    for _ in range(20):
        mid = (lo + hi) / 2
        if space.contains(cx + (outside[0] - cx) * mid, cy + (outside[1] - cy) * mid):
            lo = mid
        else:
            hi = mid
    return (cx + (outside[0] - cx) * lo, cy + (outside[1] - cy) * lo)


def smooth_route(points, radius):
    """꺾인 경로의 모서리를 둥글게 만든다 (각 꺾임을 짧은 곡선으로)."""
    pts = [np.array(p, float) for p in points]
    # 거의 붙어 있는 점은 합친다
    clean = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - clean[-1]) > 2:
            clean.append(p)
    if len(clean) < 3:
        return clean
    out = [clean[0]]
    for a, b, c in zip(clean, clean[1:], clean[2:]):
        d1, d2 = b - a, c - b
        l1, l2 = np.linalg.norm(d1), np.linalg.norm(d2)
        r = min(radius, l1 / 2, l2 / 2)
        p1, p2 = b - d1 / l1 * r, b + d2 / l2 * r
        for t in np.linspace(0, 1, 8):  # 2차 베지어 곡선
            out.append((1 - t) ** 2 * p1 + 2 * (1 - t) * t * b + t ** 2 * p2)
    out.append(clean[-1])
    return out


def draw_route(img, points, color, thickness):
    """경로를 둥근 선으로 그리고, 끝에 크고 분명한 화살촉을 붙인다 (흰 테두리로 사진 위에서도 잘 보이게)."""
    pts = smooth_route(points, thickness * 8)
    if len(pts) < 2:
        return
    tip = pts[-1]
    # 화살촉 방향: 끝에서 화살촉 길이만큼 거슬러 올라간 지점 → 끝 (짧은 마지막 조각에 흔들리지 않게)
    size = thickness * 4.5
    # 끝에서 경로를 따라 화살촉 길이만큼 거슬러 올라간 지점 = 화살촉 밑동
    i, dist, base = len(pts) - 1, 0.0, pts[0]
    while i > 0:
        seg = np.linalg.norm(pts[i] - pts[i - 1])
        if dist + seg >= size:
            base = pts[i] + (pts[i - 1] - pts[i]) * ((size - dist) / max(seg, 1e-6))
            break
        dist += seg
        i -= 1
    if np.linalg.norm(tip - base) < 1:
        return
    direction = (tip - base) / np.linalg.norm(tip - base)
    normal = np.array([-direction[1], direction[0]])
    triangle = np.array([tip, base + normal * size * 0.7, base - normal * size * 0.7], np.int32)
    # 선은 화살촉 밑동까지만 (선 끝이 화살촉 밖으로 삐져나오지 않게)
    line = np.array([*pts[:i], base], np.int32)
    outline = (255, 255, 255)
    cv2.polylines(img, [line], False, outline, thickness + 4, cv2.LINE_AA)
    cv2.fillPoly(img, [triangle], outline, cv2.LINE_AA)
    cv2.polylines(img, [triangle], True, outline, 4, cv2.LINE_AA)
    cv2.polylines(img, [line], False, color, thickness, cv2.LINE_AA)
    cv2.fillPoly(img, [triangle], color, cv2.LINE_AA)


def draw_aisles(img, aisles, alpha=0.3):
    """설정 화면용: 통로를 반투명 파란색으로 표시."""
    overlay = img.copy()
    for a in aisles:
        cv2.fillPoly(overlay, [a.polygon], BLUE)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw(frame, result, show_boxes=True, light=False):
    """칸 상태(초록=빈칸, 빨강=주차됨), 추천 칸, 입구→추천 칸 경로를 그린다. 번호판은 가린다.

    light=True: 사용자 화면용. 빈칸만 초록으로 채우고 주차된 칸은 옅은 테두리만 (차가 그대로 보이게).
    """
    img = blur_plates(frame.copy(), result.get("plates", []))
    overlay = img.copy()
    thickness = max(2, img.shape[1] // 400)
    font_scale = max(0.6, img.shape[1] / 1200)

    for s in result["spaces"]:
        occupied = result["statuses"][s.id]
        if light and occupied:
            continue
        cv2.fillPoly(overlay, [s.polygon], RED if occupied else GREEN)
    img = cv2.addWeighted(overlay, 0.42 if light else 0.35, img, 0.58 if light else 0.65, 0)

    if show_boxes:
        for b in result["boxes"]:
            x1, y1, x2, y2 = map(int, b["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), BLUE, thickness)

    # 입구 → 추천 칸 경로 (칸 번호를 가리지 않도록 번호보다 먼저, 칸 가장자리까지만)
    ex, ey = map(int, result["entrance"])
    best = next((s for s in result["spaces"] if s.id == result["recommended"]), None)
    if best is not None:
        route = result.get("routes", {}).get(best.id)
        points = route["points"] if route else [result["entrance"], best.center]
        edge = np.array(_edge_point(best, points[-2]), float)
        inside = edge + (np.array(best.center, float) - edge) * 0.45  # 화살촉이 칸 안으로 들어가게
        points = [*points[:-1], edge, tuple(inside)]
        # 멀리 있어 작게 보이는 칸은 화살표도 가늘게 (칸을 덮지 않도록 칸 폭의 40% 이하)
        sides = [np.linalg.norm(best.polygon[i] - best.polygon[(i + 1) % len(best.polygon)]) for i in range(len(best.polygon))]
        draw_route(img, points, GOLD, int(max(4, min(thickness * 2, min(sides) * 0.4))))
    cv2.circle(img, (ex, ey), thickness * 6, GOLD, -1)
    cv2.putText(img, "IN", (ex - thickness * 6, ey + thickness * 14), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, GOLD, thickness, cv2.LINE_AA)

    for s in result["spaces"]:
        occupied = result["statuses"][s.id]
        color = RED if occupied else GREEN
        cv2.polylines(img, [s.polygon], True, color, max(1, thickness // 2) if (light and occupied) else thickness, cv2.LINE_AA)
        if light and occupied:
            continue  # 주차된 칸은 번호도 생략 (차가 보이니까)
        label = s.id
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cx, cy = map(int, s.center)
        cv2.putText(img, label, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, label, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)

    if best is not None:
        sides = [np.linalg.norm(best.polygon[i] - best.polygon[(i + 1) % len(best.polygon)]) for i in range(len(best.polygon))]
        cv2.polylines(img, [best.polygon], True, GOLD, int(max(2, min(thickness * 3, min(sides) * 0.2))), cv2.LINE_AA)

    return img


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_image(data: bytes):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
