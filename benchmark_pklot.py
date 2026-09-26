"""실제 주차장 CCTV 사진(PKLot 공개 데이터셋)으로 정확도 측정.

PKLot: 브라질 대학 주차장 3곳(PUCPR, UFPR04, UFPR05)을 맑음/흐림/비 날씨에 찍은 사진과 칸별 정답.
라이선스 CC BY 4.0 — Almeida et al., "PKLot – A robust dataset for parking lot classification" (2015)
https://huggingface.co/datasets/Voxel51/PKLot

실행: python benchmark_pklot.py --per-group 10
     (주차장 3곳 × 날씨 3종류 × 10장 = 최대 90장. 사진은 data/benchmark/ 에 한 번만 내려받는다)
"""

import argparse
import collections
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2

from detector import CarDetector
from classifier import MODEL_PATH as CLASSIFIER_PATH, SpaceClassifier
from occupancy import Lot, Space, analyze, draw

BASE_URL = "https://huggingface.co/datasets/Voxel51/PKLot/resolve/main/"
CACHE = Path(__file__).parent / "data" / "benchmark"
METHODS = ["ai", "diff", "both", "cls"]


def is_train_date(moment):
    """날짜 3일 중 2일은 학습용, 1일은 시험용. 같은 날 사진이 학습과 시험에 섞이지 않게 한다."""
    return moment.date().toordinal() % 3 != 0


def download(rel_path):
    target = CACHE / rel_path
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")  # 받다 끊긴 파일을 사진으로 착각하지 않게
        for attempt in range(5):  # 서버가 연결을 끊으면 잠깐 쉬었다가 다시 시도
            try:
                urllib.request.urlretrieve(BASE_URL + rel_path, partial)
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        partial.rename(target)
    return target


def prefetch(rel_paths, workers=6):
    """여러 사진을 동시에 내려받는다."""
    missing = [p for p in set(rel_paths) if not (CACHE / p).exists()]
    if missing:
        print(f"  사진 {len(missing)}장 내려받는 중...")
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(download, missing))


def load_samples():
    rows = []
    for s in json.loads(download("samples.json").read_text())["samples"]:
        spaces = [
            (str(p["space_id"]), p["points"][0], p["occupancy_status"])
            for p in (s.get("parking_spaces") or {}).get("polylines", [])
            if p["points"] and len(p["points"][0]) >= 3  # 꼭짓점이 빠진 칸은 제외
        ]
        rows.append({
            "file": s["filepath"], "source": s["source"], "weather": s["weather"]["label"],
            "time": datetime.fromisoformat(s["parking_timestamp"]["$date"].rstrip("Z")),
            "spaces": spaces,
        })
    return rows


def pick_reference(rows):
    """완전히 빈 사진 중 오전 9시에 가장 가까운 것 (실제로는 '새벽/아침에 빈 주차장 찍기')."""
    empty = [r for r in rows if r["spaces"] and all(st == "not occupied" for _, _, st in r["spaces"])]
    return min(empty, key=lambda r: abs(r["time"].hour * 60 + r["time"].minute - 540))


def to_lot(row, image):
    h, w = image.shape[:2]
    spaces = [Space.from_points(sid, [[x * w, y * h] for x, y in pts]) for sid, pts, _ in row["spaces"]]
    return Lot(id="pklot", name=row["source"], lat=0, lng=0, image_width=w, image_height=h,
               entrance=[0, h // 2], spaces=spaces)


class Score:
    def __init__(self):
        self.tp = self.tn = self.fp = self.fn = 0

    def add(self, predicted, actual):
        if predicted and actual:
            self.tp += 1
        elif predicted:
            self.fp += 1
        elif actual:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def total(self):
        return self.tp + self.tn + self.fp + self.fn

    def __str__(self):
        acc = (self.tp + self.tn) / max(1, self.total)
        return (f"정확도 {acc:6.1%}  (빈칸을 주차됨으로 착각 {self.fp:4d}, "
                f"차를 빈칸으로 놓침 {self.fn:4d}, 전체 {self.total}칸)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-group", type=int, default=10, help="주차장×날씨 조합마다 뽑을 사진 수")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--sources", help="시험할 주차장만 (예: ufpr05)")
    parser.add_argument("--classifier", default=str(CLASSIFIER_PATH), help="칸 분류 모델 파일")
    parser.add_argument("--fixed-spaces", action="store_true",
                        help="실제 앱처럼 기준 사진의 칸 좌표 하나로 모든 사진을 판정 (카메라가 움직이면 어긋남)")
    parser.add_argument("--no-align", action="store_true", help="카메라 흔들림 자동 보정 끄기")
    parser.add_argument("--save-errors", type=int, default=3, help="틀린 칸이 많은 사진 몇 장을 그림으로 저장")
    args = parser.parse_args()
    methods = args.methods.split(",")

    print("정답 파일 불러오는 중...")
    rows = load_samples()
    rng = random.Random(args.seed)
    detector = CarDetector() if {"ai", "both"} & set(methods) else None
    if "cls" in methods and not Path(args.classifier).exists():
        print("학습된 칸 분류 모델이 없어서 cls는 건너뛰어요. (python train_classifier.py)")
        methods.remove("cls")
    classifier = SpaceClassifier(args.classifier) if "cls" in methods else None
    scores = {m: collections.defaultdict(Score) for m in methods}

    plan = []
    sources = args.sources.split(",") if args.sources else sorted({r["source"] for r in rows})
    for source in sources:
        source_rows = [r for r in rows if r["source"] == source]
        ref_row = pick_reference(source_rows)
        picked = {}
        for weather in ["sunny", "cloudy", "rainy"]:
            group = [r for r in source_rows if r["weather"] == weather and not is_train_date(r["time"])
                     and r["time"].date() != ref_row["time"].date()]
            picked[weather] = rng.sample(group, min(args.per_group, len(group)))
        plan.append((source, ref_row, picked))
    prefetch([r["file"] for _, ref, picked in plan for r in [ref, *sum(picked.values(), [])]])

    worst = []
    started = time.perf_counter()
    for source, ref_row, picked in plan:
        reference = cv2.imread(str(download(ref_row["file"])))
        fixed_lot = to_lot(ref_row, reference)
        for weather, group_rows in picked.items():
            for row in group_rows:
                frame = cv2.imread(str(download(row["file"])))
                if frame is None:
                    print(f"  읽을 수 없는 사진 건너뜀: {row['file']}")
                    continue
                lot = fixed_lot if args.fixed_spaces else to_lot(row, frame)
                lot.auto_align = not args.no_align
                truth = {sid: st == "occupied" for sid, _, st in row["spaces"] if st != "unknown"}
                for m in methods:
                    result = analyze(lot, frame, detector, mode=m, conf=args.conf, reference=reference,
                                     classifier=classifier)
                    wrong = 0
                    for sid, actual in truth.items():
                        predicted = result["statuses"][sid]
                        for key in ("전체", source, weather):
                            scores[m][key].add(predicted, actual)
                        wrong += predicted != actual
                    if m == methods[-1]:
                        worst.append((wrong, row, frame, result, truth))
        print(f"  {source} 완료 (기준 사진 {ref_row['file']} {ref_row['time']:%m-%d %H:%M})")

    print(f"\n사진 {len(worst)}장, {time.perf_counter() - started:.0f}초\n")
    for m in methods:
        print(f"[{m}]")
        for key, score in scores[m].items():
            print(f"  {key:>7}: {score}")
        print()

    out = CACHE / "errors"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    for wrong, row, frame, result, truth in sorted(worst, key=lambda x: -x[0])[:args.save_errors]:
        img = draw(frame, result, show_boxes=True)
        for sp in result["spaces"]:  # 틀린 칸에 X 표시
            if sp.id in truth and truth[sp.id] != result["statuses"][sp.id]:
                cx, cy = map(int, sp.center)
                cv2.drawMarker(img, (cx, cy), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 18, 3)
        name = f"{wrong:03d}_{row['source']}_{row['weather']}_{Path(row['file']).stem}.jpg"
        cv2.imwrite(str(out / name), img)
    print(f"틀린 칸이 많은 사진을 {out} 에 저장했어요 (노란 X = 틀린 칸)")


if __name__ == "__main__":
    main()
