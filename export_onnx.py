"""학습한 모델을 PyTorch 없이 돌릴 수 있는 ONNX로 변환한다 (다른 노트북·배포 서버용).

실행: python export_onnx.py   (ultralytics가 설치된 학습 환경에서)
결과: models/<이름>.onnx, models/<이름>.json (클래스 이름, 입력 크기)
"""

import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
# (파일 이름, 입력 크기, 사진 비율에 맞춰 입력 크기를 바꿀지)
MODELS = [("space_classifier", 64, False), ("yolo26n", 640, True)]


def main():
    for name, imgsz, dynamic in MODELS:
        pt = ROOT / "models" / f"{name}.pt"
        if not pt.exists():
            print(f"건너뜀: {pt.name} 없음")
            continue
        model = YOLO(str(pt))
        model.export(format="onnx", imgsz=imgsz, simplify=True, device="cpu", dynamic=dynamic)
        (ROOT / "models" / f"{name}.json").write_text(json.dumps(
            {"task": model.task, "imgsz": imgsz, "names": {int(k): v for k, v in model.names.items()}},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"변환 완료: {name}.onnx")


if __name__ == "__main__":
    main()
