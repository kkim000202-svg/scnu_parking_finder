"""실제 CCTV 사진 예시 주차장을 앱에 추가한다 (PKLot UFPR04, 28칸).

출처: Almeida et al., "PKLot – A robust dataset for parking lot classification" (2015), CC BY 4.0
실행: python make_pklot_demo.py
결과: data/lots/pklot_ufpr04/ 에 lot.json, empty.jpg, samples/*.jpg, labels.json
"""

import json
import random
import shutil

from benchmark_pklot import download, is_train_date, load_samples, pick_reference, prefetch, to_lot
from occupancy import LOTS_DIR

import cv2

SOURCE = "ufpr04"
LOT_ID = "pklot_ufpr04"


def main():
    rows = [r for r in load_samples() if r["source"] == SOURCE]
    reference = pick_reference(rows)
    rng = random.Random(3)
    picked = []
    for weather in ["sunny", "cloudy", "rainy"]:
        # 학습에 쓰지 않은 날짜에서, 칸이 적당히 섞여 찬 사진을 고른다
        group = [r for r in rows if r["weather"] == weather and not is_train_date(r["time"])
                 and len(r["spaces"]) >= 20
                 and 0.15 < sum(st == "occupied" for _, _, st in r["spaces"]) / len(r["spaces"]) < 0.95]
        picked += rng.sample(group, min(2, len(group)))
    prefetch([reference["file"]] + [r["file"] for r in picked])

    folder = LOTS_DIR / LOT_ID
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "samples").mkdir(parents=True)

    ref_image = cv2.imread(str(download(reference["file"])))
    lot = to_lot(reference, ref_image)
    lot.id, lot.name = LOT_ID, "실제 CCTV 예시 (브라질 UFPR 대학 주차장)"
    lot.lat, lot.lng = -25.4505, -49.2320
    lot.entrance = [ref_image.shape[1] - 40, ref_image.shape[0] - 40]
    lot.save()
    cv2.imwrite(str(lot.reference_path), ref_image)

    labels = {}
    for row in picked:
        name = f"{row['weather']}_{row['time']:%Y%m%d_%H%M}.jpg"
        shutil.copy(download(row["file"]), folder / "samples" / name)
        labels[name] = {sid: st == "occupied" for sid, _, st in row["spaces"] if st != "unknown"}
    (folder / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (folder / "SOURCE.txt").write_text(
        "PKLot dataset (UFPR04) — Almeida, P. R. L. et al., "
        "\"PKLot – A robust dataset for parking lot classification\", "
        "Expert Systems with Applications, 2015. License: CC BY 4.0\n", encoding="utf-8")
    print(f"완료: {folder} (사진 {len(picked)}장)")


if __name__ == "__main__":
    main()
