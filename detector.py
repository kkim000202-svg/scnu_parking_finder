"""YOLO로 사진 속 차량 찾기."""

from pathlib import Path

import cv2
from PIL import Image

MODEL_PATH = Path(__file__).parent / "models" / "yolo26n.pt"

# COCO 데이터셋 기준 차량 종류
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


class CarDetector:
    def __init__(self, model_path=MODEL_PATH):
        from onnx_models import OnnxDetector, available
        if available("yolo26n"):  # PyTorch 없이 (다른 노트북·배포 서버용)
            self.onnx, self.model = OnnxDetector("yolo26n"), None
        else:
            from ultralytics import YOLO  # 파일이 없으면 ultralytics가 자동으로 내려받는다
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            self.onnx, self.model = None, YOLO(str(model_path))

    def detect(self, frame, conf=0.25, vehicles_only=True):
        """차량 박스 목록: [{"box": (x1, y1, x2, y2), "label": "car", "conf": 0.87}, ...]"""
        if self.onnx:
            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            raw = self.onnx.predict(rgb, conf=conf)
        else:
            r = self.model(frame, conf=conf, verbose=False)[0]
            raw = [(*b, s, self.model.names[int(c)]) for b, s, c in
                   zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
        return [{"box": (x1, y1, x2, y2), "label": label, "conf": float(score)}
                for x1, y1, x2, y2, score, label in raw
                if not vehicles_only or label in VEHICLE_CLASSES]
