"""칸 하나를 잘라 "차 있음/빈칸"을 판단하는 직접 학습시킨 AI.

YOLO 분류 모델을 실제 주차장 CCTV 사진(PKLot)의 칸 이미지로 학습시켰다 (train_classifier.py).
기준 사진이 필요 없고, 그림자·날씨 변화에 강하다.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).parent / "models" / "space_classifier.pt"
PATCH_SIZE = 64
MARGIN = 0.15  # 칸 주변을 조금 더 잘라 차의 윤곽이 잘리지 않게 한다


def crop_space(frame, space, size=PATCH_SIZE):
    """칸을 감싸는 사각형(+여백)을 잘라 size×size로 만든다. 학습과 판정에 똑같이 쓴다."""
    h, w = frame.shape[:2]
    mx = (space.x2 - space.x1) * MARGIN
    my = (space.y2 - space.y1) * MARGIN
    x1, y1 = int(max(0, space.x1 - mx)), int(max(0, space.y1 - my))
    x2, y2 = int(min(w, space.x2 + mx)), int(min(h, space.y2 + my))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(frame[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)


def available():
    from onnx_models import available as onnx_available
    return onnx_available("space_classifier") or MODEL_PATH.exists()


class SpaceClassifier:
    def __init__(self, model_path=MODEL_PATH):
        from onnx_models import OnnxClassifier, available as onnx_available
        if onnx_available("space_classifier"):  # PyTorch 없이 (다른 노트북·배포 서버용)
            self.onnx, self.model = OnnxClassifier("space_classifier"), None
        else:
            from ultralytics import YOLO
            self.onnx, self.model = None, YOLO(str(model_path))
            self.occupied_index = next(i for i, n in self.model.names.items() if n == "occupied")

    def predict(self, frame, spaces, threshold=0.5):
        """칸별 (주차됨 여부, 주차됨 확률)."""
        if not spaces:
            return {}, {}
        patches = [crop_space(frame, s) for s in spaces]
        if self.onnx:
            # 사진은 OpenCV 형식(BGR)이라 RGB로 바꿔서 넣는다 (학습 때와 같게)
            occupied = [self.onnx.predict(Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)))["occupied"]
                        for p in patches]
        else:
            results = self.model(patches, imgsz=PATCH_SIZE, verbose=False)
            occupied = [float(r.probs.data[self.occupied_index]) for r in results]
        probs = {s.id: float(p) for s, p in zip(spaces, occupied)}
        return {sid: p >= threshold for sid, p in probs.items()}, probs
