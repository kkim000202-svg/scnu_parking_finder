# 🅿️ AI 빈자리 주차 내비

> **목적지까지는 카카오 내비처럼, 주차장 안에서는 AI가 빈칸까지 안내합니다.**

내비게이션은 주차장 입구까지만 데려다줍니다. 이 서비스는 주차장 CCTV 화면을 직접 학습시킨 AI로 분석해
지도에서 **주차장별 빈칸 수**를 보여 주고, 도착하면 **입구에서 가장 가까운 빈칸까지 가는 길**을 CCTV 화면 위에 그려 줍니다.

2026 SCNU OSS·AI 해커톤 기초 트랙 출품작 · 국립순천대학교 교내 CCTV 3곳(45칸) 적용

학교 CCTV 화면을 교내 PC의 중계 프로그램(`cctv_relay.py`)이 보내 주면 실시간으로 판정하고, 연결이 없으면 학교에서 받은 녹화 CCTV로 판정합니다.

## 주요 기능

| 기능 | 설명 |
|---|---|
| 카카오맵 내비 | 목적지·출발지 검색, 자동차 경로와 회전 안내. 지나간 경로와 안내는 사라짐 |
| AI 주차장 빈칸 수 | 지도 핀에 `P 빈칸 10/13`처럼 표시, 5초마다 새로 고침 |
| 도착 직전 확인 | 주차장 입구 10m 전에 CCTV 화면을 다시 분석해 지금 빈칸을 확인 |
| 빈칸까지 안내 | 도착하면 CCTV 화면(또는 도면) 위에 빈칸·추천 칸·입구에서 가는 길을 그림 |
| 모의 주행 / 실제 주행 | 시연용 모의 주행, 폰 GPS를 따라가는 실제 주행 (경로 이탈 시 자동 재탐색) |
| 개인정보 보호 | 번호판 자동 가림, 사진은 축소해서 전송, 카카오 REST 키는 서버에만, 접속 기록(좌표) 미저장 |
| 실시간 CCTV 수신 | `POST /api/cctv/<주차장>` — 토큰이 맞는 중계 프로그램의 사진만 받음 |

## 사용 기술 (전부 무료)

| 역할 | 기술 |
|---|---|
| 칸 분류 AI | Ultralytics YOLO26n-cls를 PKLot 데이터로 직접 학습 → ONNX 변환 (`models/space_classifier.onnx`) |
| AI 실행 | ONNX Runtime (CPU만 사용, PyTorch 불필요) |
| 영상 처리 | OpenCV, NumPy |
| 서버 | Python 3.10+, FastAPI, Uvicorn |
| 지도·검색·길찾기 | 카카오맵 JavaScript SDK, 카카오 로컬 API, 카카오모빌리티 길찾기 API |

## 내 컴퓨터에서 실행

Windows는 `run_windows.bat`, Mac은 `run_mac.command`를 더블클릭하면 처음 한 번 필요한 프로그램을 설치하고 `http://localhost:8000`을 엽니다.
카카오 키는 `.env.example`을 `.env`로 복사해 넣거나 `python setup_keys.py`로 입력합니다.
카카오 개발자 콘솔의 JavaScript SDK 도메인에 `http://localhost:8000`을 등록해야 지도가 열립니다.

> 교내 CCTV 사진(`data/lots/cctv_*/frames`, `empty.jpg`)은 개인정보 때문에 저장소에 없습니다. 그래서 저장소만 받으면 CCTV 주차장 3곳은 지도에 나오지 않고, 연습용 주차장(`demo`, `pklot_ufpr04`)만 관리자 화면에서 쓸 수 있습니다.

## 배포 방법

1. 이 폴더를 서버에 올리고 `pip install -r requirements.txt`
2. 서버 **환경변수**를 설정합니다 (키를 파일로 올리지 마세요).

| 이름 | 값 |
|---|---|
| `KAKAO_JS_KEY` | 카카오 JavaScript 키 (브라우저 지도용) |
| `KAKAO_REST_KEY` | 카카오 REST API 키 (서버 전용, 검색·길찾기) |
| `SOURCE_URL` | (선택) 공개 소스 저장소 주소 `https://...` — 화면에 "소스 코드" 링크로 표시 (AGPL) |
| `RATE_LIMIT_PER_MIN` | (선택) IP당 1분 카카오 API 호출 한도, 기본 120 |
| `CCTV_TOKEN` | 실시간 중계용 비밀 문자열. 영문·숫자로 길고 무작위로 (예: `python -c "import secrets;print(secrets.token_urlsafe(32))"`). 비워 두면 실시간 수신이 꺼짐 |

3. 카카오 개발자 콘솔 → 앱 → 플랫폼 키 → JavaScript 키 → **JavaScript SDK 도메인에 배포 주소**(`https://...`)를 추가합니다.
4. `sh start.sh` 로 실행합니다 (서버가 주는 `PORT` 사용, 접속 기록 끔).
5. 확인: `/` 지도, `/api/lots/cctv_d4` 가 200 이면 정상.

> `data/lots/cctv_*/frames/` 와 `empty.jpg` 는 학교 CCTV 화면이라 공개 저장소(.gitignore)에는 빠져 있습니다. 서버에는 이 폴더를 통째로(zip/업로드) 올려야 합니다.

## 학교 CCTV를 실시간으로 연결하기

학교 CCTV는 교내망 안에 있어서 밖의 서버가 직접 볼 수 없습니다. **교내 PC 한 대**에서 `cctv_relay.py` 를 실행해 CCTV 화면을 2초마다 서버로 보냅니다.

```
[학교 CCTV 녹화기] ──교내망──▶ [교내 PC: cctv_relay.py] ──https──▶ [서버: AI 판정] ──▶ 사용자 폰
```

```bash
pip install opencv-python-headless httpx
export CCTV_TOKEN=서버와_같은_토큰                       # Windows: set CCTV_TOKEN=...
export CCTV_SOURCE="rtsp://아이디:비번@CCTV주소:554/..."   # 학교 전산 담당에게 받은 주소
python cctv_relay.py --server https://우리주소 --camera cctv_d4
```

- 비밀번호가 명령 기록에 남지 않게 토큰과 CCTV 주소는 환경변수로 넣습니다. 화면 출력에서는 아이디·비밀번호를 가립니다.
- 서버 주소는 `https://` 만 허용합니다 (시험용 localhost 제외).
- `--camera` 는 `data/lots/` 아래 주차장 폴더 이름(`cctv_a1`, `cctv_b7`, `cctv_d4`)입니다.
- 연결 시험: `--source 녹화영상.avi` 로 파일을 실시간처럼 보낼 수 있습니다.

## 정확도

### 실제 CCTV 사진 (PKLot)

브라질 대학 주차장 3곳의 실제 CCTV 사진 중, **학습에 쓰지 않은 날짜**의 사진 250장(칸 12,160개)으로 측정했습니다.
맑음·흐림·비 날씨가 골고루 섞여 있습니다.

| 판정 방식 | 정확도 |
|---|---|
| YOLO 차량 인식 | 44.3% |
| 기준 사진 비교 | 83.1% |
| YOLO + 기준 사진 비교 | 83.2% |
| **칸 분류 AI (직접 학습)** | **99.7%** |

**처음 보는 주차장에서도 되나요?** 주차장 2곳으로만 학습하고 나머지 1곳으로 시험했습니다.

| 학습에서 뺀 주차장 | 정확도 |
|---|---|
| PUCPR (100칸) | 97.6% |
| UFPR04 (28칸) | 99.9% |
| UFPR05 (40칸) | 98.6% |

**카메라가 움직이면?** 실제 앱처럼 기준 사진의 칸 좌표 하나로 모든 날짜의 사진을 판정했습니다.
UFPR04는 날짜마다 카메라 방향이 최대 100px 넘게 달라지는 주차장입니다.

| | 전체 | UFPR04 |
|---|---|---|
| 흔들림 보정 끔 | 97.8% | 88.3% |
| **흔들림 보정 켬** | **99.3%** | **97.3%** |

> 틀린 칸 중에는 사진에 차가 분명히 있는데 데이터셋 정답이 "빈칸"으로 잘못 적힌 경우도 있어서, 실제 정확도는 이보다 조금 더 높을 수 있습니다.

학습·측정 스크립트(`train_classifier.py`, `benchmark_pklot.py`)는 개발 저장소에 있습니다.

### 순천대 교내 CCTV

학교에서 받은 녹화 CCTV(카메라 3대, 약 5분씩)로 주차장 3곳, 45칸을 등록했습니다.
정답 라벨은 없어서 사람이 화면을 보고 확인했으며, 판정 결과와 일치했습니다.

| 주차장 | 칸 | 비고 |
|---|---|---|
| 공대 3호관 뒤 | 13 | 멀리 작게 보이는 칸 제외 |
| 박물관 뒤 (옥상 카메라) | 25 | 나무에 가린 칸 제외. 전기차 칸의 초록 페인트 때문에 판정 기준을 0.85로 높임 |
| 진리관 앞 | 7 | 소나무에 가린 칸 제외 |

## 개발용 파일

| 파일 | 용도 |
|---|---|
| `app.py` | 관리자 화면 (Streamlit): 주차장 등록, 칸·입구·통로 그리기 — `run_admin_windows.bat` / `run_admin_mac.command` |
| `train_classifier.py` | 칸 분류 AI 학습 (PKLot 자동 다운로드), `requirements-train.txt` 필요 |
| `benchmark_pklot.py`, `evaluate.py` | 정확도 측정 |
| `export_onnx.py` | 학습한 모델을 ONNX로 변환 |
| `make_demo_data.py`, `make_pklot_demo.py` | 연습용 주차장 만들기 |
| `mini.py`, `mini_*.pdf` | 책상 위 미니어처 주차장 (마커 원근 보정) |
| `tools_lines.py` | CCTV 화면에서 칸 선을 찾는 보조 도구 |

## 폴더 구조

```
web/server.py            서버 (카카오 API 중계, 빈칸 판정{", CCTV 수신" if live else ""})
web/static/              카카오맵 화면 (index.html, app.js, style.css)
occupancy.py             칸 판정, 흔들림 보정, 통로 경로, 번호판 가림, 결과 그리기
classifier.py            칸 분류 AI 실행
onnx_models.py           ONNX 모델 실행
mini.py                  (서버가 불러오는 보조 모듈)
models/                  AI 모델 (ONNX)
data/lots/<주차장>/       lot.json(칸·입구·통로 좌표), empty.jpg, frames/(CCTV 사진)
start.sh                 서버 시작{chr(10)+"cctv_relay.py            교내 CCTV 중계 프로그램" if live else ""}
```

## 한계

- {"실시간 연결 코드는 준비됐지만 학교 CCTV 주소를 받지 못해, 지금은 녹화 영상으로 동작합니다." if live else "학교 CCTV 실시간 연결이 안 돼 녹화 영상(아침 5분)으로 판정합니다. 차 출입이 적어 빈칸 수 변화가 작습니다."}
- 주차장 안에서 차 위치는 GPS 오차(5~10m) 때문에 칸 단위로 추적하지 못합니다.
- 칸 분류 AI는 낮 시간 사진으로 학습했습니다. 야간 화면은 검증하지 않았습니다.
- 번호판 가림은 규칙 기반이라 놓칠 수 있습니다. 대신 사진을 축소해서 보내 번호판·사람을 알아볼 수 없는 해상도로 제한합니다.

## 데이터 출처

- **PKLot** — Almeida, P. R. L., Oliveira, L. S., Britto Jr., A. S., Silva Jr., E. J., Koerich, A. L.,
  "PKLot – A robust dataset for parking lot classification", *Expert Systems with Applications*, 2015.
  [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/). 칸 분류 AI 학습, 정확도 측정, 앱의 실제 CCTV 예시 주차장에 사용했습니다.

## 라이선스

[AGPL-3.0](LICENSE). 이 프로젝트가 사용하는 Ultralytics YOLO가 AGPL-3.0이기 때문입니다.
