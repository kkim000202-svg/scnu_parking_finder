"""카카오맵 중심 사용자 화면 + API 서버.

실행: python -m uvicorn web.server:app --port 8000   (프로젝트 폴더에서)
화면: http://localhost:8000

- 지도·검색·길찾기: 카카오 API (키는 .env 의 KAKAO_JS_KEY, KAKAO_REST_KEY)
- 빈칸 판정: 직접 학습시킨 칸 분류 AI (occupancy.py, classifier.py) — 카카오 키 없이도 동작
- REST 키는 브라우저에 보내지 않고 이 서버에서만 쓴다.
"""

import base64
import hmac
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import httpx
from dotenv import load_dotenv
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from classifier import SpaceClassifier, available as classifier_available  # noqa: E402
import mini  # noqa: E402
from occupancy import MODE_CLS, _edge_point, analyze, blur_plates, draw, imread, list_lots, load_lot, open_video  # noqa: E402

JS_KEY = REST_KEY = ""


def load_keys():
    """카카오 키를 .env에서 읽는다. 키가 없으면 요청마다 다시 읽어서, 나중에 .env를 만들어도 서버 재시작 없이 동작한다."""
    global JS_KEY, REST_KEY
    if JS_KEY and REST_KEY:
        return
    load_dotenv(ROOT / ".env", override=True)
    JS_KEY = os.environ.get("KAKAO_JS_KEY", "").strip().strip("'\"")
    REST_KEY = os.environ.get("KAKAO_REST_KEY", "").strip().strip("'\"")


load_keys()
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="AI 빈자리 주차 안내")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# 화면을 다른 주소(Genspark 호스팅)에 올릴 때: 그 주소에서 오는 요청을 허용한다.
# ALLOWED_ORIGINS="https://a.pages.dev,https://b.com" 처럼 쉼표로. 비워 두면 모든 주소 허용 (쿠키·로그인이 없어 안전)
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins or ["*"], allow_methods=["GET"], allow_headers=["*"])


@app.get("/health")
def health():
    """배포 확인·서버 깨우기용."""
    return {"ok": True}

# 카카오 API를 부르는 주소는 한 사람(IP)이 1분에 RATE_LIMIT번까지만 (무료 쿼터를 남이 다 써 버리지 않게)
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", "60"))
_hits = {}  # IP → 최근 1분 호출 시각들
_LIMITED = ("/api/search", "/api/route", "/api/parking")


@app.middleware("http")
async def rate_limit(request, call_next):
    if request.url.path.startswith(_LIMITED) and not request.query_params.get("ours_only"):
        ip = (request.headers.get("x-forwarded-for", "").split(",")[0].strip()
              or (request.client.host if request.client else "?"))
        now = time.time()
        recent = [t for t in _hits.get(ip, []) if now - t < 60]
        if len(recent) >= RATE_LIMIT:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "요청이 너무 많아요. 1분 뒤에 다시 시도해 주세요."}, status_code=429)
        recent.append(now)
        _hits[ip] = recent
        if len(_hits) > 5000:  # 오래된 IP 정리
            for k in [k for k, v in _hits.items() if not v or now - v[-1] > 60]:
                _hits.pop(k, None)
    return await call_next(request)


# ---------- 화면 ----------

@app.get("/", response_class=HTMLResponse)
def index():
    load_keys()
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    source_url = os.environ.get("SOURCE_URL", "").strip()
    if not source_url.startswith("https://"):
        source_url = ""
    return html.replace("__KAKAO_JS_KEY__", JS_KEY).replace("__SOURCE_URL__", source_url)


# ---------- 카카오 API 중계 (REST 키를 브라우저에 노출하지 않기 위해) ----------

def _kakao(url, params):
    load_keys()
    if not REST_KEY:
        raise HTTPException(503, "카카오 REST API 키가 없어요. .env 파일에 KAKAO_REST_KEY를 넣어 주세요.")
    try:
        r = httpx.get(url, params=params, headers={"Authorization": f"KakaoAK {REST_KEY}"}, timeout=10)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"카카오 서버에 연결하지 못했어요 ({exc.__class__.__name__}).")
    if r.status_code == 401:
        raise HTTPException(502, "카카오 키가 올바르지 않아요. REST API 키를 다시 확인해 주세요.")
    if r.status_code == 403:
        raise HTTPException(502, "이 카카오 API를 쓸 권한이 없어요. 앱 설정에서 해당 기능을 켜 주세요.")
    if r.status_code != 200:
        print(f"[카카오 API 오류] {r.status_code} {r.text[:200]}", flush=True)  # 자세한 내용은 서버 로그에만
        raise HTTPException(502, f"카카오 API 오류 ({r.status_code}). 잠시 후 다시 시도해 주세요.")
    return r.json()


@app.get("/api/search")
def search(q: str = Query(..., min_length=1, max_length=50), x: float | None = None, y: float | None = None):
    """장소 이름으로 검색 (출발지·목적지 입력)."""
    params = {"query": q, "size": 8}
    if x and y:
        params.update(x=x, y=y, sort="accuracy")
    data = _kakao("https://dapi.kakao.com/v2/local/search/keyword.json", params)
    return [{"name": d["place_name"], "address": d.get("road_address_name") or d.get("address_name"),
             "x": float(d["x"]), "y": float(d["y"]), "category": d.get("category_group_name", "")}
            for d in data.get("documents", [])]


@app.get("/api/route")
def route(ox: float, oy: float, dx: float, dy: float):
    """자동차 길찾기: 경로 좌표, 거리·시간, 회전 안내."""
    data = _kakao("https://apis-navi.kakaomobility.com/v1/directions",
                  {"origin": f"{ox},{oy}", "destination": f"{dx},{dy}", "priority": "RECOMMEND"})
    routes = data.get("routes") or []
    if not routes or routes[0].get("result_code") != 0:
        msg = routes[0].get("result_msg") if routes else "경로를 찾지 못했어요."
        raise HTTPException(404, f"경로를 찾지 못했어요: {msg}")
    r = routes[0]
    path, guides = [], []
    for section in r.get("sections", []):
        for road in section.get("roads", []):
            v = road["vertexes"]
            path += [[v[i + 1], v[i]] for i in range(0, len(v), 2)]  # [위도, 경도]
        for g in section.get("guides", []):
            guides.append({"lat": g["y"], "lng": g["x"], "text": g.get("guidance", ""),
                           "name": g.get("name", ""), "distance": g.get("distance", 0), "type": g.get("type", 0)})
    return {"distance": r["summary"]["distance"], "duration": r["summary"]["duration"],
            "path": path, "guides": guides}


@app.get("/api/parking")
def parking(x: float, y: float, radius: int = Query(1000, ge=100, le=5000), ours_only: bool = False):
    """목적지 주변 주차장: 카카오 장소 검색 결과 + 우리 AI가 있는 주차장. ours_only=true면 우리 것만 (빈칸 새로 고침용)."""
    # 지도 위치(입구 좌표)가 등록된 주차장만 지도에 보여준다 (연습용 데모 주차장 제외)
    ours = [lot_summary(i) for i in list_lots() if load_lot(i).entrance_latlng]
    ours = [o for o in ours if o and _distance_m(y, x, o["lat"], o["lng"]) <= max(radius, 1500)]
    # 같은 자리에 실시간 카메라 주차장이 켜져 있으면 녹화 사진 주차장은 숨긴다
    ours = [o for o in ours if o["live"] or not any(
        p["live"] and _distance_m(o["lat"], o["lng"], p["lat"], p["lng"]) < 40 for p in ours)]
    others = []
    if REST_KEY and not ours_only:
        data = _kakao("https://dapi.kakao.com/v2/local/search/category.json",
                      {"category_group_code": "PK6", "x": x, "y": y, "radius": radius, "sort": "distance", "size": 15})
        for d in data.get("documents", []):
            lat, lng = float(d["y"]), float(d["x"])
            if any(_distance_m(lat, lng, o["lat"], o["lng"]) < 40 for o in ours):
                continue  # 우리 주차장과 같은 곳이면 우리 쪽 정보를 쓴다
            others.append({"name": d["place_name"], "lat": lat, "lng": lng,
                           "address": d.get("road_address_name") or d.get("address_name"),
                           "distance": int(d.get("distance") or 0), "url": d.get("place_url") if str(d.get("place_url") or "").startswith(("https://", "http://")) else None})
    return {"ours": ours, "others": others}


def _distance_m(lat1, lng1, lat2, lng2):
    import math
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ---------- 우리 주차장: 빈칸 판정 ----------

_classifier = None
_live = {}  # lot_id → (받은 시각, 카메라 사진)  — 미니어처 카메라 페이지용
_cctv = {}  # 카메라 ID(주차장 폴더) → (받은 시각, CCTV 사진)  — 교내 중계 프로그램(cctv_relay.py)이 보냄
LIVE_TIMEOUT = 30  # 카메라 사진이 이 시간(초) 넘게 안 오면 꺼진 것으로 본다
_cache = {}  # lot_id → (시각, 결과)
_lock = threading.Lock()
CACHE_SECONDS = 10
MIN_FRESH_SECONDS = 2  # fresh=true 요청도 이 간격보다 자주 다시 판정하지 않는다
FRAME_SECONDS = 2  # frames/ 사진 한 장이 녹화 영상의 몇 초인지


def _get_classifier():
    global _classifier
    if _classifier is None and classifier_available():
        _classifier = SpaceClassifier()
    return _classifier


def _current_frame(lot):
    """지금 이 주차장을 찍은 사진. 실제 서비스에서는 CCTV의 최신 프레임, 시연에서는 녹화 영상·샘플 사진."""
    # 녹화 CCTV를 2초 간격 사진으로 저장해 둔 주차장: 지금 시각에 맞는 장면 (영상처럼 계속 돌아감)
    frame_dir = (lot.folder.parent / lot.frames_from if lot.frames_from else lot.folder) / "frames"
    frames = sorted(frame_dir.glob("*.jpg")) if frame_dir.exists() else []
    if frames:
        return imread(frames[int(time.time() / FRAME_SECONDS) % len(frames)])
    videos = sorted((lot.folder / "videos").glob("*")) if (lot.folder / "videos").exists() else []
    if videos:
        cap = open_video(videos[0])
        count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 10
        if count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((time.time() * fps) % count))
            ok, frame = cap.read()
            cap.release()
            if ok:
                return frame
    samples = sorted((lot.folder / "samples").glob("*.jpg"))
    return imread(samples[0]) if samples else None


def _live_summary_result(lot_id, lot):
    """실시간 카메라 주차장: 가장 최근 카메라 사진으로 판정. 마커가 잠깐 가려지면 직전 결과를 쓴다."""
    got = _live.get(lot_id)
    if not got or time.time() - got[0] > LIVE_TIMEOUT:
        return None
    cached = _cache.get(lot_id)
    if cached and cached[0] >= got[0]:
        return cached
    result, top, _ = mini.analyze_live(lot, got[1])
    if result is None:
        return cached if cached and got[0] - cached[0] < 10 else None
    cached = (got[0], result, top)
    _cache[lot_id] = cached
    return cached


_history = {}  # lot_id → 최근 판정들 (영상 주차장의 깜빡임 방지)


def _smooth(lot_id, lot, result):
    """녹화 영상 주차장: 칸마다 최근 3번 판정 중 2번 이상이 '주차됨'일 때만 주차됨으로 본다."""
    if not ((lot.folder / "videos").exists() or (lot.folder / "frames").exists() or lot.frames_from):
        return result
    hist = _history.setdefault(lot_id, [])
    hist.append(result["statuses"])
    del hist[:-3]
    statuses = {k: sum(h.get(k, False) for h in hist) * 2 > len(hist) for k in result["statuses"]}
    from occupancy import recommend
    return {**result, "statuses": statuses, "empty_count": sum(1 for v in statuses.values() if not v),
            "recommended": recommend(result["spaces"], statuses, result["entrance"], result.get("routes"))}


def _live_cctv_frame(lot, lot_id):
    """교내 중계 프로그램이 보낸 최신 CCTV 사진 (끊긴 지 LIVE_TIMEOUT초가 넘으면 None → 녹화 영상으로 돌아감)."""
    got = _cctv.get(lot.frames_from or lot_id)
    return got if got and time.time() - got[0] <= LIVE_TIMEOUT else None


def _pick_entrance(lot, result, frame, approach):
    """차가 다가오는 지점(approach: 위도, 경도)에 가장 가까운 입구를 골라 주차장 안 경로를 다시 계산한다.

    칸 판정은 그대로 두고 경로·추천 칸만 바꾼다 (입구마다 가장 가까운 빈칸이 다르므로).
    """
    if not lot.entrances or approach is None:
        return result, (lot.entrances or [{}])[0].get("id")
    from occupancy import plan_routes, recommend
    ent = min(lot.entrances, key=lambda e: _distance_m(approach[0], approach[1], *e["latlng"]))
    h, w = frame.shape[:2]
    px = (ent["px"][0] * w / lot.image_width, ent["px"][1] * h / lot.image_height)
    size = (w, h)
    routes = plan_routes(result["spaces"], result["aisles"], px, size, exclude_spaces=True) \
        or plan_routes(result["spaces"], result["aisles"], px, size)
    best = recommend(result["spaces"], result["statuses"], px, routes)
    return {**result, "entrance": px, "routes": routes, "recommended": best}, ent["id"]


def lot_summary(lot_id, with_image=False, fresh=False, approach=None):
    lot = load_lot(lot_id)
    latlng = lot.entrance_latlng
    is_live = bool(lot.live)
    with _lock:
        if lot.live:
            cached = _live_summary_result(lot_id, lot)
            if cached is None:
                return None
        else:
            cached = _cache.get(lot_id)
        live = None if lot.live else _live_cctv_frame(lot, lot_id)
        if live is not None:
            is_live = True
            if not cached or cached[0] < live[0]:  # 새 CCTV 사진이 들어왔을 때만 다시 판정
                clf = _get_classifier()
                if clf is None or not lot.spaces:
                    return None
                result = _smooth(lot_id, lot, analyze(lot, live[1], mode=MODE_CLS, classifier=clf))
                cached = (live[0], result, live[1])
                _cache[lot_id] = cached
        elif not lot.live and (not cached or time.time() - cached[0] > (MIN_FRESH_SECONDS if fresh else CACHE_SECONDS)):
            frame = _current_frame(lot)
            clf = _get_classifier()
            if frame is None or clf is None or not lot.spaces:
                return None
            result = analyze(lot, frame, mode=MODE_CLS, classifier=clf)
            result = _smooth(lot_id, lot, result)
            cached = (time.time(), result, frame)
            _cache[lot_id] = cached
    _, result, frame = cached
    result, entrance_id = _pick_entrance(lot, result, frame, approach)
    summary = {
        "id": lot_id, "name": lot.name, "lat": lot.lat, "lng": lot.lng,
        "entrance": latlng or [lot.lat, lot.lng],
        "empty": result["empty_count"], "total": result["total"], "recommended": result["recommended"],
        "camera_shift": round(result.get("camera_shift", 0)), "updated": int(cached[0]),
        "live": is_live, "entrance_id": entrance_id,
    }
    if with_image:
        img = draw(frame, result, show_boxes=False, light=True)
        h, w = img.shape[:2]
        img = cv2.resize(img, (900, int(h * 900 / w)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        summary["image"] = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        summary["plan"] = _plan(frame, result)
        # 표시 없는 CCTV 원본 (화면에서 칸·경로를 직접 겹쳐 그린다)
        raw = cv2.resize(blur_plates(frame.copy(), result.get("plates", [])), (1000, int(h * 1000 / w)))
        ok, buf = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 80])
        summary["photo"] = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    return summary


def _plan(frame, result):
    """주차장 도면용 데이터: 칸 모양·상태, 통로, 입구, 입구→추천 칸 경로 (사진 픽셀 좌표)."""
    h, w = frame.shape[:2]
    pts = lambda poly: [[round(float(x)), round(float(y))] for x, y in poly.reshape(-1, 2)]  # noqa: E731
    spaces = [{"id": s.id, "points": pts(s.polygon), "center": [round(float(v)) for v in s.center],
               "occupied": bool(result["statuses"][s.id])} for s in result["spaces"]]
    route = None
    best = next((s for s in result["spaces"] if s.id == result["recommended"]), None)
    if best is not None:
        r = result.get("routes", {}).get(best.id)
        points = r["points"] if r else [result["entrance"], best.center]
        route = [[round(float(x)), round(float(y))] for x, y in [*points[:-1], _edge_point(best, points[-2])]]
    return {"width": w, "height": h, "spaces": spaces,
            "aisles": [pts(a.polygon) for a in result["aisles"]],
            "entrance": [round(float(v)) for v in result["entrance"]],
            "route": route, "recommended": result["recommended"]}


@app.get("/api/lots/{lot_id}")
def lot_detail(lot_id: str, fresh: bool = False, alat: float | None = None, alng: float | None = None):
    """fresh=true: 캐시를 쓰지 않고 지금 CCTV 화면을 새로 분석 (주차장 10m 전에 호출).
    alat, alng: 차가 주차장으로 다가오는 지점 → 그쪽 입구에서 빈칸까지 경로를 그린다."""
    if lot_id not in list_lots():
        raise HTTPException(404, "없는 주차장이에요.")
    approach = (alat, alng) if alat is not None and alng is not None else None
    summary = lot_summary(lot_id, with_image=True, fresh=fresh, approach=approach)
    if summary is None:
        raise HTTPException(503, "이 주차장의 사진이나 AI 모델이 없어요.")
    return summary


@app.post("/api/cctv/{camera_id}")
async def cctv_upload(camera_id: str, request: Request):
    """교내 중계 프로그램(cctv_relay.py)이 보내는 실시간 CCTV 사진.

    camera_id = CCTV 사진을 쓰는 주차장 폴더 이름 (카메라 하나를 여러 주차장이 같이 쓰면 frames_from이 가리키는 곳).
    아무나 가짜 사진을 못 보내게, .env 또는 환경변수의 CCTV_TOKEN과 같은 값을 X-CCTV-Token 헤더로 보내야 한다.
    CCTV_TOKEN이 없으면 이 기능은 꺼져 있다.
    """
    load_dotenv(ROOT / ".env")
    token = os.environ.get("CCTV_TOKEN", "").strip()
    if not token:
        raise HTTPException(403, "실시간 CCTV 연결이 꺼져 있어요. 서버에 CCTV_TOKEN을 설정해 주세요.")
    if not hmac.compare_digest(request.headers.get("X-CCTV-Token", "").encode("latin-1", "replace"), token.encode("utf-8")):
        raise HTTPException(401, "CCTV_TOKEN이 맞지 않아요.")
    if camera_id not in list_lots():
        raise HTTPException(404, "없는 주차장이에요.")
    limit = 8 * 1024 * 1024
    if int(request.headers.get("content-length") or 0) > limit:
        raise HTTPException(413, "사진이 너무 커요 (8MB 이하).")
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise HTTPException(413, "사진이 너무 커요 (8MB 이하).")
    body = bytes(body)
    frame = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "사진을 읽지 못했어요.")
    if frame.shape[1] > 1600:  # 너무 큰 사진은 줄여서 보관 (메모리 절약)
        frame = cv2.resize(frame, (1600, round(frame.shape[0] * 1600 / frame.shape[1])))
    with _lock:
        _cctv[camera_id] = (time.time(), frame)
    users = [i for i in list_lots() if i == camera_id or load_lot(i).frames_from == camera_id]
    return {"ok": True, "lots": users, "size": [frame.shape[1], frame.shape[0]]}


@app.get("/api/config")
def config():
    return {"has_js_key": bool(JS_KEY), "has_rest_key": bool(REST_KEY)}
