#!/bin/bash
cd "$(dirname "$0")"
if [ ! -f .venv/installed_v2.txt ]; then
  echo "=== 처음 실행: 필요한 프로그램을 설치해요 (5~10분, 인터넷 필요) ==="
  python3 -m venv .venv || { echo "[오류] Python 3가 필요해요: https://www.python.org/downloads/"; read -r; exit 1; }
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt || { echo "[오류] 설치 실패. 인터넷을 확인하세요."; read -r; exit 1; }
  echo ok > .venv/installed_v2.txt
fi
echo "=== 앱을 켜는 중이에요. 잠시 후 브라우저가 열려요. 끝내려면 Ctrl+C ==="
(sleep 20; open http://localhost:8000) &
.venv/bin/python -m uvicorn web.server:app --port 8000
