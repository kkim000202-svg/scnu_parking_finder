"""판정 방식별 정확도 측정 (발표 자료용).

준비: data/lots/<주차장ID>/samples/ 에 사진, labels.json 에 정답
      {"사진이름.jpg": {"1": true, "2": false, ...}}   true = 주차됨
실행: python evaluate.py demo
"""

import json
import sys

import cv2

from classifier import MODEL_PATH as CLASSIFIER_PATH, SpaceClassifier
from detector import CarDetector
from occupancy import imread
from occupancy import MODE_AI, MODE_BOTH, MODE_CLS, MODE_DIFF, analyze, load_lot


def main(lot_id):
    lot = load_lot(lot_id)
    labels = json.loads((lot.folder / "labels.json").read_text(encoding="utf-8"))
    detector = CarDetector()
    classifier = SpaceClassifier() if CLASSIFIER_PATH.exists() else None

    print(f"주차장: {lot.name} / 사진 {len(labels)}장 / 칸 {len(lot.spaces)}개\n")
    modes = [(MODE_AI, "YOLO"), (MODE_DIFF, "기준 사진 비교"), (MODE_BOTH, "YOLO+비교")]
    if classifier:
        modes.append((MODE_CLS, "칸 분류 AI"))
    for mode, title in modes:
        correct = total = 0
        wrong = []
        for name, answer in labels.items():
            frame = imread(lot.folder / "samples" / name)
            result = analyze(lot, frame, detector, mode=mode, classifier=classifier)
            for space_id, occupied in answer.items():
                total += 1
                if result["statuses"].get(space_id) == occupied:
                    correct += 1
                else:
                    wrong.append(f"{name}#{space_id}")
        print(f"[{title:>8}] 정확도 {correct / total:6.1%} ({correct}/{total})"
              + (f"  틀린 칸: {', '.join(wrong[:6])}{' ...' if len(wrong) > 6 else ''}" if wrong else ""))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo")
