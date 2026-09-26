"""교내 CCTV 중계 프로그램 — 학교 안 PC에서 실행한다.

학교 CCTV는 교내망 안에 있어서 밖에 있는 서버가 직접 볼 수 없다.
이 프로그램이 CCTV 화면을 읽어서 몇 초마다 한 장씩 서버로 보낸다. 서버는 받은 사진으로 바로 빈칸을 판정하고,
사진이 30초 넘게 안 오면 자동으로 녹화 영상으로 돌아간다.

실행 예:
  # 학교 CCTV (녹화기 RTSP 주소는 학교 전산 담당에게 받는다. 비밀번호가 명령 기록에 남지 않게 환경변수로)
  export CCTV_TOKEN=서버와_같은_토큰
  export CCTV_SOURCE="rtsp://아이디:비번@192.168.0.10:554/ch1"
  python cctv_relay.py --server https://우리주소 --camera cctv_d4

  # 녹화 영상 파일로 연결 시험 (실시간처럼 영상 속도대로 보낸다)
  python cctv_relay.py --server http://localhost:8000 --camera cctv_d4 --source "D4뒷주차장.avi"

  # 노트북 카메라
  python cctv_relay.py --server http://localhost:8000 --camera cctv_d4 --source 0

토큰: 서버의 CCTV_TOKEN과 같은 값을 --token 또는 환경변수 CCTV_TOKEN으로 준다.
카메라 ID(--camera): data/lots/ 아래 주차장 폴더 이름.
카메라 하나가 주차장 여러 곳을 비추면, 다른 주차장이 lot.json 의 frames_from 으로 이 폴더를 가리키게 해 두면 같이 바뀐다.
"""

import argparse
import os
from urllib.parse import urlsplit
import threading
import time
from pathlib import Path

import cv2
import httpx


def masked(source):
    """화면에 찍을 때 RTSP 주소 속 아이디·비밀번호를 가린다."""
    parts = urlsplit(str(source))
    if parts.password or parts.username:
        return str(source).replace(parts.netloc, "***@" + (parts.hostname or "") + (f":{parts.port}" if parts.port else ""))
    return str(source)


class LatestFrame:
    """CCTV를 계속 읽어서 가장 최근 장면만 들고 있는다.

    RTSP는 읽지 않으면 영상이 쌓여서 몇 분 전 화면이 나오므로, 별도 스레드가 계속 읽어 버린다.
    영상 파일은 실제 재생 속도에 맞춰 읽어서 실시간처럼 흉내 낸다 (끝나면 처음부터 다시).
    """

    def __init__(self, source):
        self.source = int(source) if str(source).isdigit() else source
        self.is_file = isinstance(self.source, str) and Path(self.source).exists()
        self.frame, self.stamp, self.error = None, 0.0, None
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _open(self):
        if self.is_file and not str(self.source).isascii():
            # OpenCV는 한글 경로 영상을 못 여는 경우가 있어 프로젝트의 우회 함수를 쓴다
            from occupancy import open_video
            return open_video(Path(self.source))
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self):
        while not self._stop:
            cap = self._open()
            if not cap.isOpened():
                self.error = f"CCTV를 열지 못했어요: {masked(self.source)}"
                time.sleep(5)
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 15
            self.error = None
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    if self.is_file:  # 영상 끝 → 처음부터
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.error = "CCTV 연결이 끊겼어요. 다시 연결할게요."
                    break
                self.frame, self.stamp = frame, time.time()
                if self.is_file:
                    time.sleep(1 / fps)
            cap.release()
            time.sleep(2)

    def stop(self):
        self._stop = True


def main():
    ap = argparse.ArgumentParser(description="교내 CCTV 화면을 빈자리 주차 내비 서버로 보낸다.")
    ap.add_argument("--server", required=True, help="서버 주소 (예: https://a5.scnuoss.net)")
    ap.add_argument("--camera", required=True, help="주차장 폴더 이름 (예: cctv_d4)")
    ap.add_argument("--source", default=os.environ.get("CCTV_SOURCE", ""),
                    help="RTSP 주소, 영상 파일, 또는 카메라 번호(0). 비밀번호가 든 주소는 환경변수 CCTV_SOURCE 권장")
    ap.add_argument("--token", default=os.environ.get("CCTV_TOKEN", ""), help="서버의 CCTV_TOKEN")
    ap.add_argument("--interval", type=float, default=2.0, help="몇 초마다 보낼지 (기본 2초)")
    ap.add_argument("--width", type=int, default=1600, help="보내기 전에 줄일 가로 크기 (기본 1600px)")
    args = ap.parse_args()
    if not args.token:
        ap.error("--token 또는 환경변수 CCTV_TOKEN이 필요해요.")
    if not args.source:
        ap.error("--source 또는 환경변수 CCTV_SOURCE가 필요해요.")
    server = urlsplit(args.server)
    if server.scheme != "https" and server.hostname not in ("localhost", "127.0.0.1", "::1"):
        ap.error("서버 주소는 https:// 여야 해요 (토큰과 CCTV 사진이 그대로 보이지 않게). localhost 시험만 http 허용.")

    url = f"{args.server.rstrip('/')}/api/cctv/{args.camera}"
    reader = LatestFrame(args.source)
    client = httpx.Client(timeout=15)
    sent, last_stamp, last_error = 0, 0.0, None
    print(f"중계 시작: {masked(args.source)} → {url} ({args.interval}초마다). 끝내려면 Ctrl+C")
    try:
        while True:
            t0 = time.time()
            if reader.error:
                if reader.error != last_error:  # 같은 오류는 한 번만
                    print(reader.error)
                    last_error = reader.error
            elif reader.frame is not None and reader.stamp != last_stamp:
                frame, last_stamp = reader.frame, reader.stamp
                h, w = frame.shape[:2]
                if w > args.width:
                    frame = cv2.resize(frame, (args.width, round(h * args.width / w)))
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    continue
                last_error = None
                try:
                    r = client.post(url, content=buf.tobytes(),
                                    headers={"Content-Type": "image/jpeg", "X-CCTV-Token": args.token})
                    try:
                        body = r.json()
                    except ValueError:
                        body = {}
                    if r.status_code == 200:
                        sent += 1
                        lots = ", ".join(body.get("lots", []) if isinstance(body, dict) else [])
                        print(f"\r보냄 {sent}장 · {time.strftime('%H:%M:%S')} · 주차장: {lots}   ", end="", flush=True)
                    else:
                        detail = body.get("detail", r.text[:100]) if isinstance(body, dict) else r.text[:100]
                        print(f"\n서버가 거절했어요 ({r.status_code}): {detail}")
                        if r.status_code in (401, 403, 404):
                            break
                except httpx.HTTPError as exc:
                    print(f"\n서버에 보내지 못했어요 ({exc.__class__.__name__}). 다시 시도할게요.")
                except Exception as exc:  # 예상 못 한 오류에도 중계는 계속
                    print(f"\n오류: {exc.__class__.__name__}. 계속 시도할게요.")
            time.sleep(max(0.0, args.interval - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\n중계를 멈췄어요. 30초 뒤 서버는 녹화 영상으로 돌아가요.")
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
