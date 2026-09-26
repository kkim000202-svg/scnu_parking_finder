"""PyTorch 없이 ONNX Runtime으로 AI 모델을 돌린다 (배포 서버 메모리 제한 700MB에 맞추기 위해).

모델 파일(.onnx)과 클래스 이름(.json)은 training/export_onnx.py가 만든다.
전처리는 학습 때(Ultralytics)와 똑같이 맞춰야 결과가 같게 나온다.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parent / "models"


def _session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2  # 작은 서버에서 CPU·메모리를 덜 쓰게
    options.enable_cpu_mem_arena = False  # 메모리를 미리 크게 잡아두지 않게 (서버 제한 700MB)
    options.enable_mem_pattern = False
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _meta(name):
    meta = json.loads((MODEL_DIR / f"{name}.json").read_text(encoding="utf-8"))
    meta["names"] = {int(k): v for k, v in meta["names"].items()}
    return meta


class OnnxDetector:
    """YOLO 탐지 모델 (사진 비율에 맞춰 입력 크기가 바뀌는 ONNX)."""

    def __init__(self, name):
        self.session = _session(MODEL_DIR / f"{name}.onnx")
        meta = _meta(name)
        self.names, self.size = meta["names"], meta["imgsz"]
        self.input = self.session.get_inputs()[0].name

    def _letterbox(self, img: Image.Image, stride=32):
        """비율을 유지해 긴 변을 size로 줄이고, 32의 배수가 되도록 회색(114)으로 가운데를 채운다.
        (Ultralytics LetterBox(auto=True)와 같게 — 640×640으로 채우는 것보다 작은 물건을 더 잘 찾는다)"""
        arr = np.asarray(img.convert("RGB"))
        h, w = arr.shape[:2]
        r = min(self.size / h, self.size / w)
        nw, nh = round(w * r), round(h * r)
        pad_w, pad_h = ((self.size - nw) % stride) / 2, ((self.size - nh) % stride) / 2
        if (nw, nh) != (w, h):
            arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
        left, right = round(pad_w - 0.1), round(pad_w + 0.1)
        arr = cv2.copyMakeBorder(arr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        blob = arr.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return blob, r, left, top

    def _decode(self, out, conf, iou=0.7, max_det=300):
        """모델 출력 → [x1, y1, x2, y2, 확신도, 클래스 번호] (입력 사진 좌표).
        - 고정 크기 출력 [300, 6+]: 겹친 후보가 이미 제거된 최종 결과
        - 가변 크기 출력 [4+클래스 수(+마스크 32), 후보 수]: 후보 전체라서 겹친 후보를 직접 걸러낸다(NMS)"""
        if out.shape[-1] >= 6 and out.shape[0] == 300 and out.shape[1] < out.shape[0]:
            return out[out[:, 4] >= conf, :6]
        pred = out.T  # [후보 수, 4 + 클래스 수 (+ 마스크 계수)]
        nc = len(self.names)
        scores = pred[:, 4:4 + nc]
        cls = scores.argmax(1)
        score = scores[np.arange(len(pred)), cls]
        keep = score >= conf
        pred, cls, score = pred[keep], cls[keep], score[keep]
        if not len(pred):
            return np.zeros((0, 6), np.float32)
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1)
        idx = cv2.dnn.NMSBoxesBatched(boxes.tolist(), score.tolist(), cls.tolist(), conf, iou)
        idx = np.array(idx, dtype=int).reshape(-1)[:max_det]
        return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, score, cls], 1)[idx]

    def predict(self, img: Image.Image, conf=0.25):
        """[(x1, y1, x2, y2, 확신도, 클래스 이름)] — 원본 사진 픽셀 좌표."""
        blob, r, left, top = self._letterbox(img)
        out = self._decode(self.session.run(None, {self.input: blob})[0][0], conf)
        w, h = img.size
        found = []
        for x1, y1, x2, y2, score, cls in out:
            x1, x2 = np.clip([(x1 - left) / r, (x2 - left) / r], 0, w)
            y1, y2 = np.clip([(y1 - top) / r, (y2 - top) / r], 0, h)
            found.append((float(x1), float(y1), float(x2), float(y2), float(score), self.names[int(cls)]))
        return found


class OnnxClassifier:
    """YOLO26 분류 모델. 출력 [1, 클래스 수] 확률."""

    def __init__(self, name):
        self.session = _session(MODEL_DIR / f"{name}.onnx")
        meta = _meta(name)
        self.names, self.size = meta["names"], meta["imgsz"]
        self.input = self.session.get_inputs()[0].name

    def predict(self, img: Image.Image):
        """짧은 변을 size로 줄이고 가운데를 size×size로 자른다 (Ultralytics classify_transforms와 같게)."""
        img = img.convert("RGB")
        w, h = img.size
        scale = self.size / min(w, h)
        img = img.resize((max(self.size, round(w * scale)), max(self.size, round(h * scale))), Image.BILINEAR)
        w, h = img.size
        left, top = round((w - self.size) / 2), round((h - self.size) / 2)
        img = img.crop((left, top, left + self.size, top + self.size))
        blob = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        probs = self.session.run(None, {self.input: blob})[0][0]
        return {self.names[i]: float(p) for i, p in enumerate(probs)}


def available(name):
    return (MODEL_DIR / f"{name}.onnx").exists() and (MODEL_DIR / f"{name}.json").exists()
