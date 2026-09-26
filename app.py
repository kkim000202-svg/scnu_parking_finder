"""빈자리 주차 안내 웹 앱.

실행: streamlit run app.py
"""

import threading
import time
from pathlib import Path
from urllib.parse import quote

import cv2
import folium
import numpy as np
import streamlit as st
from streamlit_folium import st_folium
from streamlit_image_coordinates import streamlit_image_coordinates

from classifier import SpaceClassifier, available as classifier_available
from detector import CarDetector
from occupancy import imread, imwrite, open_video
from occupancy import (
    MODE_AI, MODE_BOTH, MODE_CLS, MODE_DIFF, Lot, Space, StatusSmoother,
    analyze, apply_smoothing, decode_image, draw, draw_aisles, list_lots, load_lot,
    plan_routes, scale_rects, to_rgb,
)

st.set_page_config(page_title="빈자리 주차 안내", page_icon="🅿️", layout="wide")

MODES = {}
if classifier_available():
    MODES["칸 분류 AI (추천, 실제 CCTV 사진으로 직접 학습)"] = MODE_CLS
MODES.update({
    "YOLO 차량 인식 + 기준 사진 비교": MODE_BOTH,
    "YOLO 차량 인식만": MODE_AI,
    "기준 사진 비교만": MODE_DIFF,
})
SETUP_WIDTH = 1000  # 설정 화면으로 보내는 사진 폭(px). 화면에는 칸 폭에 맞춰 늘어난다


# ---------- 공용 자원 ----------

@st.cache_resource(show_spinner="AI 모델을 불러오는 중...")
def get_detector():
    return CarDetector()


@st.cache_resource(show_spinner="칸 분류 AI를 불러오는 중...")
def get_classifier():
    return SpaceClassifier()


VIDEO_TYPES = ["mp4", "mov", "m4v", "avi"]


@st.cache_resource(show_spinner=False)
def open_cameras():
    """열어 둔 카메라들 {source: (VideoCapture, 잠금)}. 모든 접속자가 함께 쓴다."""
    return {}


def open_camera(source):
    """source: 카메라 번호(0, 1, ...) 또는 IP 카메라 앱의 영상 주소(http://..., rtsp://...)"""
    cameras = open_cameras()
    if source not in cameras or not cameras[source][0].isOpened():
        # 여러 사람이 동시에 접속해도 같은 카메라를 번갈아 읽도록 잠금을 같이 둔다
        cameras[source] = (cv2.VideoCapture(source), threading.Lock())
    return cameras[source]


def release_cameras():
    """카메라를 실제로 닫는다 (캐시만 비우면 카메라 불이 계속 켜져 있다)."""
    cameras = open_cameras()
    for cap, lock in cameras.values():
        with lock:
            cap.release()
    cameras.clear()


def read_camera(source):
    cap, lock = open_camera(source)
    if not cap.isOpened():
        release_cameras()
        return None
    with lock:
        for _ in range(2):  # 쌓여 있던 예전 프레임 버리기
            cap.grab()
        ok, frame = cap.read()
    return frame if ok else None


@st.cache_resource(show_spinner=False)
def open_video(path):
    cap = open_video(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, count, threading.Lock()


def read_video(path, started_at):
    """녹화 영상을 CCTV처럼 실제 시간에 맞춰 재생한다. 끝나면 처음부터 반복."""
    cap, fps, count, lock = open_video(path)
    if not cap.isOpened() or count <= 0:
        return None, 0, 0
    position = (time.time() - started_at) % (count / fps)
    with lock:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(position * fps))
        ok, frame = cap.read()
    return (frame if ok else None), position, count / fps


# ---------- 사이드바 ----------

lots = list_lots()
if not lots:
    st.error("주차장 설정이 없어요. 먼저 `python make_demo_data.py` 를 실행하세요.")
    st.stop()

# 위젯이 만들어진 뒤에는 값을 바꿀 수 없으므로, 새로 만든 주차장 선택은 다음 실행 때 반영
if "pending_lot_id" in st.session_state:
    st.session_state["lot_id"] = st.session_state.pop("pending_lot_id")

with st.sidebar:
    st.title("🅿️ 빈자리 주차 안내")
    lot_id = st.selectbox("주차장", lots, key="lot_id",
                          format_func=lambda i: load_lot(i).name)
    lot = load_lot(lot_id)

    st.divider()
    st.subheader("분석 설정")
    mode = MODES[st.radio("판정 방식", list(MODES))]
    conf = st.slider("AI 신뢰도 기준", 0.05, 0.9, 0.25, 0.05,
                     help="낮추면 차를 더 많이 찾지만 잘못 찾는 경우도 늘어요.")
    show_boxes = st.checkbox("AI가 찾은 차 박스 표시", value=True)
    lot.auto_align = st.checkbox(
        "카메라 흔들림 자동 보정", value=lot.auto_align, key=f"align_{lot.id}",
        help="카메라가 조금 움직여도 기준 사진과 맞춰서 칸 위치를 자동으로 옮겨요. 기준 사진이 있어야 동작해요.")

    with st.expander("기준 사진 비교 민감도"):
        st.caption("실제 주차장이나 미니어처에 맞춰 조절하세요. "
                   "아래 **칸별 판정 상세**의 '바뀐 면적'을 보면서 맞추면 쉬워요.")
        lot.pixel_threshold = st.slider(
            "색 차이 기준", 10, 120, lot.pixel_threshold, 5, key=f"px_{lot.id}",
            help="픽셀 색이 이만큼 이상 달라져야 '바뀐 픽셀'로 셉니다. 그림자에 속으면 올리세요.")
        lot.changed_ratio = st.slider(
            "바뀐 면적 기준(%)", 5, 80, int(lot.changed_ratio * 100), 5, key=f"ratio_{lot.id}",
            help="칸 안에서 바뀐 픽셀이 이 비율 이상이면 주차됨. 빈칸을 주차됨으로 착각하면 올리세요.") / 100
        if st.button("이 주차장에 저장", width="stretch"):
            lot.save()
            st.toast("민감도를 저장했어요.")

    st.divider()
    st.subheader("카메라")
    camera_input = st.text_input(
        "카메라 (번호 또는 주소)", "0",
        help="Mac에 연결된 카메라는 번호(내장 0, 아이폰 연속성 카메라는 보통 1)를, "
             "폰의 IP 카메라 앱을 쓰면 앱에 나오는 영상 주소(http://... 또는 rtsp://...)를 적어요.").strip()
    camera = int(camera_input) if camera_input.isdigit() else camera_input
    refresh = st.slider("실시간 분석 간격(초)", 0.5, 10.0, 2.0, 0.5)
    stable_frames = st.slider(
        "깜빡임 방지 (연속 확인 횟수)", 1, 5, 2,
        help="같은 판정이 이 횟수만큼 연속으로 나와야 칸 상태를 바꿔요. "
             "차를 넣는 손이 잠깐 가려도 화면이 흔들리지 않아요.")
    if st.button("카메라 연결 끊기"):
        release_cameras()
        st.toast("카메라를 껐어요.")

tab_guide, tab_setup = st.tabs(["🚗 빈자리 안내", "🛠️ 주차장 설정"])


# ---------- 빈자리 안내 ----------

def get_smoother(source_key):
    """실시간 모드용 깜빡임 방지 상태. 주차장·영상 출처·설정이 바뀌면 새로 만든다."""
    key = (lot.id, source_key, stable_frames)
    if st.session_state.get("smoother_key") != key:
        st.session_state["smoother_key"] = key
        st.session_state["smoother"] = StatusSmoother(stable_frames)
    return st.session_state["smoother"]


def show_details(result):
    with st.expander("칸별 판정 상세"):
        rows = []
        for sp in result["spaces"]:
            ratio = result["diff_ratios"].get(sp.id)
            ai = result["ai_statuses"].get(sp.id)
            prob = result["cls_probs"].get(sp.id)
            rows.append({
                "칸": sp.id,
                "AI": "-" if ai is None else ("차 있음" if ai else "없음"),
                "바뀐 면적": "-" if ratio is None else f"{ratio:.0%}",
                "학습 AI 주차 확률": "-" if prob is None else f"{prob:.0%}",
                "최종": "🟥 주차됨" if result["statuses"][sp.id] else "🟩 빈칸",
            })
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(f"'바뀐 면적'이 {lot.changed_ratio:.0%} 이상이면 주차됨으로 판정해요.")


def show_result(frame, smoother=None):
    if not lot.spaces:
        st.info("아직 주차 칸이 없어요. **🛠️ 주차장 설정** 탭에서 칸을 추가하세요.")
        return
    if mode in (MODE_DIFF, MODE_BOTH) and lot.reference_image() is None:
        st.warning("기준 사진(빈 주차장)이 없어서 비교 판정은 건너뛰어요.")

    started = time.perf_counter()
    detector = get_detector() if mode in (MODE_AI, MODE_BOTH) else None
    classifier = get_classifier() if mode == MODE_CLS else None
    result = analyze(lot, frame, detector, mode=mode, conf=conf, classifier=classifier)
    if smoother is not None:
        result = apply_smoothing(result, smoother)
    elapsed = time.perf_counter() - started

    col_img, col_info = st.columns([3, 1.3])
    with col_img:
        st.image(to_rgb(draw(frame, result, show_boxes)), width="stretch")
    with col_info:
        st.metric("남은 칸", f"{result['empty_count']} / {result['total']}")
        if result["recommended"]:
            st.success(f"### 👉 {result['recommended']}번 칸\n입구에서 가장 가까운 빈칸이에요.")
        else:
            st.error("### 빈칸이 없어요\n다른 주차장을 이용해 주세요.")
        empty = [k for k, v in result["statuses"].items() if not v]
        full = [k for k, v in result["statuses"].items() if v]
        st.markdown(f"🟩 **빈칸:** {', '.join(empty) or '없음'}")
        st.markdown(f"🟥 **주차됨:** {', '.join(full) or '없음'}")
        if result["camera_shift"]:
            st.caption(f"📐 카메라가 최대 {result['camera_shift']:.0f}px 움직여서 칸 위치를 자동으로 맞췄어요.")
        st.caption(f"AI가 찾은 차 {len(result['boxes'])}대 · 분석 {elapsed * 1000:.0f}ms")
    show_details(result)


with tab_guide:
    source = st.radio("사진 가져오기", ["샘플 사진", "사진 올리기", "녹화 영상", "실시간 카메라"],
                      horizontal=True)
    if source != "실시간 카메라" and st.session_state.pop("camera_in_use", False):
        release_cameras()  # 실시간 카메라 화면에서 나가면 카메라도 끈다

    if source == "샘플 사진":
        samples = sorted((lot.folder / "samples").glob("*.jpg"))
        if not samples:
            st.info("이 주차장에는 샘플 사진이 없어요.")
        else:
            chosen = st.select_slider("샘플", samples, format_func=lambda p: p.stem)
            show_result(imread(chosen))

    elif source == "사진 올리기":
        st.caption("휴대폰으로 이 페이지에 접속했다면 버튼을 눌러 바로 사진을 찍을 수 있어요.")
        uploaded = st.file_uploader("주차장 사진", type=["jpg", "jpeg", "png"])
        if uploaded:
            show_result(decode_image(uploaded.getvalue()))

    elif source == "녹화 영상":
        st.caption("폰으로 미리 찍어 둔 주차장 영상을 CCTV처럼 재생하면서 분석해요. "
                   "카메라 연결이 필요 없어서 발표장에서도 안전해요.")
        video_dir = lot.folder / "videos"
        new_video = st.file_uploader("영상 추가", type=VIDEO_TYPES, key=f"video_upload_{lot.id}")
        if new_video and st.session_state.get("video_done") != new_video.file_id:
            st.session_state["video_done"] = new_video.file_id
            video_dir.mkdir(parents=True, exist_ok=True)
            (video_dir / Path(new_video.name).name).write_bytes(new_video.getvalue())
        videos = sorted(p for p in video_dir.glob("*") if p.suffix.lower().lstrip(".") in VIDEO_TYPES) \
            if video_dir.exists() else []
        if not videos:
            st.info("영상이 없어요. 위에서 영상을 추가하세요.")
        else:
            video = str(st.selectbox("영상", videos, format_func=lambda p: p.name))
            start_key = f"video_start_{video}"
            if start_key not in st.session_state or st.button("⏮ 처음부터"):
                st.session_state[start_key] = time.time()

            @st.fragment(run_every=refresh)
            def video_view():
                frame, position, duration = read_video(video, st.session_state[start_key])
                if frame is None:
                    st.error("영상을 읽을 수 없어요. 아이폰 영상이라면 설정 → 카메라 → 포맷에서 "
                             "'높은 호환성'으로 바꿔 다시 찍어 보세요.")
                    return
                st.caption(f"▶ {position:.1f}초 / {duration:.1f}초")
                show_result(frame, get_smoother(video))

            video_view()

    else:
        st.session_state["camera_in_use"] = True
        st.caption("주차장을 위에서 비추도록 카메라를 고정하세요. "
                   "처음 실행하면 macOS가 카메라 권한을 물어볼 수 있어요.")

        @st.fragment(run_every=refresh)
        def live_view():
            frame = read_camera(camera)
            if frame is None:
                st.error(f"카메라 '{camera}'를 열 수 없어요. 사이드바에서 카메라 번호나 주소를 확인하세요.")
                return
            show_result(frame, get_smoother(camera_input))

        live_view()

    st.divider()
    col_map, col_nav = st.columns([3, 1.3])
    with col_map:
        m = folium.Map(location=[lot.lat, lot.lng], zoom_start=17)
        folium.Marker([lot.lat, lot.lng], tooltip=lot.name,
                      icon=folium.Icon(color="green", icon="car", prefix="fa")).add_to(m)
        st_folium(m, height=320, use_container_width=True, returned_objects=[], key="map")
    with col_nav:
        st.subheader("📍 주차장까지 길안내")
        st.write(lot.name)
        name = quote(lot.name)
        st.link_button("네이버 지도 앱으로 길안내",
                       f"nmap://route/car?dlat={lot.lat}&dlng={lot.lng}"
                       f"&dname={name}&appname=scnu.parkingfinder",
                       type="primary", width="stretch")
        st.link_button("카카오맵으로 길안내",
                       f"https://map.kakao.com/link/to/{name},{lot.lat},{lot.lng}",
                       width="stretch")
        st.caption("네이버 지도 버튼은 휴대폰에 네이버 지도 앱이 있어야 열려요.")


# ---------- 주차장 설정 ----------

def is_new_event(value, key):
    """클릭 컴포넌트는 새로고침될 때마다 마지막 값을 다시 돌려주므로 처음 본 이벤트만 처리한다."""
    if not value:
        return False
    stamp = value.get("unix_time")
    if st.session_state.get(key) == stamp:
        return False
    st.session_state[key] = stamp
    return True


def set_reference(target, image):
    """기준 사진을 바꾸고, 크기가 달라지면 기존 칸 좌표를 비율대로 옮긴다."""
    h, w = image.shape[:2]
    sx, sy = w / target.image_width, h / target.image_height
    target.spaces = scale_rects(target.spaces, sx, sy)
    target.aisles = scale_rects(target.aisles, sx, sy)
    target.entrance = [int(target.entrance[0] * sx), int(target.entrance[1] * sy)]
    target.image_width, target.image_height = w, h
    target.folder.mkdir(parents=True, exist_ok=True)
    imwrite(target.reference_path, image)
    target.save()


with tab_setup:
    with st.expander("➕ 새 주차장 만들기"):
        with st.form("new_lot"):
            new_id = st.text_input("영문 ID (폴더 이름)", placeholder="scnu_mini")
            new_name = st.text_input("이름", placeholder="미니어처 주차장")
            c1, c2 = st.columns(2)
            new_lat = c1.number_input("위도", value=34.9690, format="%.6f")
            new_lng = c2.number_input("경도", value=127.4798, format="%.6f")
            if st.form_submit_button("만들기"):
                if not new_id.isidentifier() or not new_id.isascii():
                    st.error("ID는 영문, 숫자, _ 만 쓸 수 있어요.")
                elif new_id in lots:
                    st.error("이미 있는 ID예요.")
                else:
                    Lot(id=new_id, name=new_name or new_id, lat=new_lat, lng=new_lng,
                        image_width=1280, image_height=720, entrance=[40, 360]).save()
                    st.session_state["pending_lot_id"] = new_id
                    st.rerun()

    st.subheader(f"{lot.name} 설정")

    with st.expander("기본 정보"):
        with st.form("lot_info"):
            name = st.text_input("이름", lot.name)
            c1, c2 = st.columns(2)
            lat = c1.number_input("위도", value=lot.lat, format="%.6f")
            lng = c2.number_input("경도", value=lot.lng, format="%.6f")
            if st.form_submit_button("저장"):
                lot.name, lot.lat, lot.lng = name, lat, lng
                lot.save()
                st.rerun()

    st.markdown("#### 1. 기준 사진 (차가 한 대도 없는 빈 주차장)")
    st.caption("칸 좌표를 그리는 바탕이 되고, '기준 사진 비교' 판정에도 쓰여요. "
               "카메라 위치를 고정한 뒤 찍어야 해요.")
    c1, c2 = st.columns(2)
    ref_upload = c1.file_uploader("사진 올리기", type=["jpg", "jpeg", "png"], key="ref_upload")
    if ref_upload and st.session_state.get("ref_done") != ref_upload.file_id:
        st.session_state["ref_done"] = ref_upload.file_id
        set_reference(lot, decode_image(ref_upload.getvalue()))
        st.rerun()
    if c2.button(f"📷 카메라 '{camera_input}'로 지금 찍기"):
        frame = read_camera(camera)
        release_cameras()  # 한 장만 찍으면 되니 바로 끈다
        if frame is None:
            c2.error("카메라를 열 수 없어요.")
        else:
            set_reference(lot, frame)
            st.rerun()

    reference = lot.reference_image()
    if reference is None:
        st.info("기준 사진을 먼저 올리거나 찍어 주세요.")
        st.stop()

    st.markdown("#### 2. 칸, 입구, 통로 표시")
    tool = st.radio("도구", ["칸 추가: 칸의 한쪽 모서리에서 반대쪽 모서리까지 드래그",
                             "기울어진 칸 추가: 칸의 꼭짓점 4개를 차례로 클릭",
                             "입구 지정: 입구 위치를 클릭",
                             "통로 추가: 차가 다니는 길을 사각형으로 드래그 (여러 개를 이어 붙여도 돼요)"])

    route_target = None
    if lot.aisles and lot.spaces:
        route_target = st.selectbox("경로 미리보기", [None] + [s.id for s in lot.spaces],
                                    format_func=lambda i: "보지 않기" if i is None else f"입구 → {i}번 칸")
    routes = plan_routes(lot.spaces, lot.aisles, lot.entrance, (lot.image_width, lot.image_height))
    preview_result = {
        "spaces": lot.spaces, "entrance": lot.entrance, "routes": routes,
        "statuses": {s.id: False for s in lot.spaces}, "recommended": route_target, "boxes": [],
    }
    preview = draw(draw_aisles(reference, lot.aisles), preview_result, show_boxes=False)
    pending_key = f"pending_points_{lot.id}"
    pending = st.session_state.setdefault(pending_key, [])
    for x, y in pending:  # 기울어진 칸: 지금까지 찍은 꼭짓점
        cv2.circle(preview, (int(x), int(y)), max(4, lot.image_width // 150), (0, 200, 255), -1)
    if len(pending) > 1:
        cv2.polylines(preview, [np.array(pending, np.int32)], False, (0, 200, 255), 2)
    display = cv2.resize(preview, (SETUP_WIDTH, int(lot.image_height * SETUP_WIDTH / lot.image_width)))

    col_canvas, col_list = st.columns([3, 1])
    with col_canvas:
        value = streamlit_image_coordinates(to_rgb(display), click_and_drag=True, width="stretch",
                                            key=f"canvas_{lot.id}", cursor="crosshair")
    if is_new_event(value, f"last_event_{lot.id}"):
        # 화면에 보이는 크기 기준 좌표 → 기준 사진 픽셀 좌표
        scale = lot.image_width / value["width"]
        x1, y1 = value["x1"] * scale, value["y1"] * scale
        x2, y2 = value["x2"] * scale, value["y2"] * scale
        if tool.startswith("입구"):
            lot.entrance = [int(x2), int(y2)]
            lot.save()
            st.rerun()
        elif tool.startswith("기울어진"):
            pending.append([int(x2), int(y2)])
            if len(pending) == 4:
                next_id = str(max((int(s.id) for s in lot.spaces if s.id.isdigit()), default=0) + 1)
                lot.spaces.append(Space.from_points(next_id, pending))
                st.session_state[pending_key] = []
                lot.save()
            st.rerun()
        elif abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
            clamp_x = lambda v: int(min(max(v, 0), lot.image_width))
            clamp_y = lambda v: int(min(max(v, 0), lot.image_height))
            rect = (clamp_x(min(x1, x2)), clamp_y(min(y1, y2)), clamp_x(max(x1, x2)), clamp_y(max(y1, y2)))
            if tool.startswith("통로"):
                lot.aisles.append(Space(f"a{len(lot.aisles) + 1}", *rect))
            else:
                next_id = str(max((int(s.id) for s in lot.spaces if s.id.isdigit()), default=0) + 1)
                lot.spaces.append(Space(next_id, *rect))
            lot.save()
            st.rerun()
        else:
            st.toast("너무 작아요. 한쪽 모서리에서 반대쪽 모서리까지 드래그하세요.")

    with col_list:
        if tool.startswith("기울어진"):
            st.info(f"꼭짓점 {len(pending)}/4개")
            if pending and st.button("찍은 점 취소", width="stretch"):
                st.session_state[pending_key] = []
                st.rerun()
        st.metric("칸 수", len(lot.spaces))
        if lot.spaces:
            target = st.selectbox("지울 칸", [s.id for s in lot.spaces])
            if st.button("선택한 칸 지우기", width="stretch"):
                lot.spaces = [s for s in lot.spaces if s.id != target]
                lot.save()
                st.rerun()
            if st.button("칸 전체 지우기", width="stretch"):
                lot.spaces = []
                lot.save()
                st.rerun()

        st.divider()
        st.metric("통로 수", len(lot.aisles),
                  help="통로가 없으면 입구에서 직선 거리로 가장 가까운 칸을 추천해요.")
        if lot.aisles:
            if st.button("마지막 통로 지우기", width="stretch"):
                lot.aisles.pop()
                lot.save()
                st.rerun()
            if st.button("통로 전체 지우기", width="stretch"):
                lot.aisles = []
                lot.save()
                st.rerun()
        st.caption("변경 사항은 자동 저장돼요.")
