@echo off
chcp 65001 > nul
cd /d "%~dp0"
title AI 빈자리 주차 안내

set PY=
where python >nul 2>nul && set PY=python
if not defined PY (where py >nul 2>nul && set PY=py)
if not defined PY (
  echo.
  echo [오류] Python이 설치되어 있지 않아요.
  echo https://www.python.org/downloads/ 에서 Python 3.12를 설치하세요.
  echo 설치 첫 화면에서 "Add python.exe to PATH"를 꼭 체크하세요.
  echo.
  pause
  exit /b 1
)

if not exist ".venv\installed_v2.txt" (
  echo.
  echo === 처음 실행: 필요한 프로그램을 설치해요. 5~10분 걸리고 인터넷이 필요해요 ===
  echo.
  %PY% -m venv .venv
  if errorlevel 1 (
    echo [오류] 가상환경을 만들지 못했어요.
    pause
    exit /b 1
  )
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [오류] 설치에 실패했어요. 인터넷 연결을 확인하고 다시 실행하세요.
    pause
    exit /b 1
  )
  echo ok > ".venv\installed_v2.txt"
)

echo.
echo === 앱을 켜는 중이에요. 잠시 후 브라우저가 열려요. ===
echo === 처음 켤 때는 20초쯤 걸려요. 브라우저가 하얗게 뜨면 새로고침 F5 ===
echo === 끝내려면 이 창을 닫으세요. ===
echo.
start "" /min cmd /c "timeout /t 20 > nul & start http://localhost:8000"
".venv\Scripts\python.exe" -m uvicorn web.server:app --port 8000
pause
