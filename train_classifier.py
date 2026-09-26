"""칸 분류 AI 학습.

1) PKLot 사진에서 칸마다 이미지를 잘라 '빈칸/주차됨' 폴더에 모은다.
2) YOLO 분류 모델(yolo26n-cls)을 그 이미지로 학습시켜 models/space_classifier.pt 로 저장한다.

학습에는 '학습용 날짜'의 사진만 쓰고, benchmark_pklot.py는 '시험용 날짜'의 사진만 써서
처음 보는 날의 사진으로 공정하게 평가한다.

실행: python train_classifier.py --per-group 40 --epochs 15
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2

from benchmark_pklot import CACHE, download, is_train_date, load_samples, prefetch, to_lot
from classifier import MODEL_PATH, PATCH_SIZE, crop_space

def build_patches(rows, per_group, seed, patches):
    rng = random.Random(seed)
    if patches.exists():
        shutil.rmtree(patches)
    counts = {}
    plan = []
    for source in sorted({r["source"] for r in rows}):
        for weather in ["sunny", "cloudy", "rainy"]:
            group = [r for r in rows if r["source"] == source and r["weather"] == weather
                     and is_train_date(r["time"])]
            plan.append((source, weather, rng.sample(group, min(per_group, len(group)))))
    prefetch([r["file"] for _, _, picked in plan for r in picked])

    for source, weather, picked in plan:
        for i, row in enumerate(picked):
            # 학습용 날짜 안에서도 10장 중 1장은 학습 중 검증(val)용으로 떼어 둔다
            split = "val" if i % 10 == 0 else "train"
            frame = cv2.imread(str(download(row["file"])))
            if frame is None:
                print(f"  읽을 수 없는 사진 건너뜀: {row['file']}")
                continue
            lot = to_lot(row, frame)
            status = {sid: st for sid, _, st in row["spaces"]}
            for space in lot.spaces:
                if status[space.id] == "unknown":
                    continue
                label = "occupied" if status[space.id] == "occupied" else "empty"
                folder = patches / split / label
                folder.mkdir(parents=True, exist_ok=True)
                name = f"{source}_{Path(row['file']).stem}_{space.id}.jpg"
                cv2.imwrite(str(folder / name), crop_space(frame, space))
                counts[(split, label)] = counts.get((split, label), 0) + 1
        print(f"  {source}/{weather}: 사진 {len(picked)}장")
    for key, n in sorted(counts.items()):
        print(f"  {key[0]:>5}/{key[1]:<8} {n}장")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-group", type=int, default=40, help="주차장×날씨 조합마다 쓸 사진 수")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-patches", action="store_true", help="이미 만든 칸 이미지를 그대로 사용")
    parser.add_argument("--holdout", help="이 주차장은 학습에서 빼기 (처음 보는 주차장 성능 시험용)")
    parser.add_argument("--output", default=str(MODEL_PATH))
    args = parser.parse_args()

    name = f"space_classifier_no_{args.holdout}" if args.holdout else "space_classifier"
    patches = CACHE / "patches" / name
    if not args.skip_patches:
        print("칸 이미지 만드는 중...")
        rows = [r for r in load_samples() if r["source"] != args.holdout]
        build_patches(rows, args.per_group, args.seed, patches)

    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH.parent / "yolo26n-cls.pt"))
    model.train(data=str(patches), imgsz=PATCH_SIZE, epochs=args.epochs, batch=256,
                device="mps", project=str(CACHE / "runs"), name=name, exist_ok=True,
                fliplr=0.5, hsv_v=0.5, plots=False, verbose=False)
    best = CACHE / "runs" / name / "weights" / "best.pt"
    shutil.copy(best, args.output)
    print(f"학습 완료: {args.output}")


if __name__ == "__main__":
    main()
