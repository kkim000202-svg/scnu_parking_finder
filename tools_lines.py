"""CCTV 한 장면에서 주차 칸 나누기 도우미 (개발용).

한 줄(row)의 윗선·아랫선을 꺾은선으로 주면, 그 사이에서 흰 칸막이 선을 자동으로 찾아
칸 다각형을 만든다. 결과는 사람이 눈으로 확인하고 lot.json에 저장한다.
"""

import cv2
import numpy as np


def _interp(poly, x):
    xs, ys = zip(*poly)
    return float(np.interp(x, xs, ys))


def white_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # 주변보다 확 밝고 색이 없는 곳 = 흰 페인트 (그늘에서도 잡히게 주변 대비로 판단)
    L = lab[:, :, 0].astype(np.float32)
    local = cv2.blur(L, (41, 41))
    return ((L - local > 22) & (hsv[:, :, 1] < 70)).astype(np.uint8)


def _runs(mask, top, bottom, f, x0, x1, min_w=3, max_w=40):
    xs = np.arange(int(x0), int(x1))
    vals = []
    for x in xs:
        y = _interp(top, x) + f * (_interp(bottom, x) - _interp(top, x))
        yi = int(round(y))
        vals.append(mask[max(0, yi - 2):yi + 3, x].max() if 0 <= yi < mask.shape[0] else 0)
    vals = np.array(vals)
    runs, start = [], None
    for i, v in enumerate(vals):
        if v and start is None:
            start = i
        if (not v or i == len(vals) - 1) and start is not None:
            end = i if not v else i + 1
            if min_w <= end - start <= max_w:
                runs.append(xs[start] + (end - start) / 2)
            start = None
    return runs


def find_dividers(img, top, bottom, x0, x1, f1=0.15, f2=0.85, max_shift=120, min_gap=40):
    """줄 안의 칸막이 선들: [(x_at_f1, x_at_f2), ...] (왼쪽부터)."""
    m = white_mask(img)
    a = _runs(m, top, bottom, f1, x0, x1)
    b = _runs(m, top, bottom, f2, x0, x1)
    pairs, used = [], set()
    for xa in a:
        cands = [(abs(xb - xa), j, xb) for j, xb in enumerate(b) if j not in used and abs(xb - xa) <= max_shift]
        if cands:
            _, j, xb = min(cands)
            used.add(j)
            pairs.append((xa, xb))
    pairs.sort()
    out = []
    for p in pairs:  # 너무 붙은 건(두 줄 선 등) 하나로
        if out and abs(p[0] - out[-1][0]) < min_gap:
            continue
        out.append(p)
    return out


def divider_line(top, bottom, pair, f1=0.15, f2=0.85):
    """(x_at_f1, x_at_f2) → 윗선·아랫선과 만나는 두 점."""
    xa, xb = pair
    pts = []
    for f_target in (0.0, 1.0):
        x = xa + (xb - xa) * (f_target - f1) / (f2 - f1)
        for _ in range(3):  # 선이 기울어 있으니 x가 바뀌면 y도 조금 바뀐다 → 몇 번 반복
            y = _interp(top, x) + f_target * (_interp(bottom, x) - _interp(top, x))
        pts.append([x, y])
    return pts


def spaces_from_dividers(top, bottom, pairs, start_id=1, skip=()):
    """이웃한 칸막이 선 사이 = 칸 하나. skip: 칸이 아닌 틈(차로 등)의 순번."""
    lines = [divider_line(top, bottom, p) for p in pairs]
    spaces, n = [], start_id
    for i in range(len(lines) - 1):
        if i in skip:
            continue
        (t1, b1), (t2, b2) = lines[i], lines[i + 1]
        spaces.append((str(n), [[round(t1[0]), round(t1[1])], [round(t2[0]), round(t2[1])],
                                [round(b2[0]), round(b2[1])], [round(b1[0]), round(b1[1])]]))
        n += 1
    return spaces


def preview(img, spaces, path, extra_lines=()):
    out = img.copy()
    for sid, pts in spaces:
        cv2.polylines(out, [np.int32(pts)], True, (0, 255, 0), 3)
        c = np.mean(pts, axis=0).astype(int)
        cv2.putText(out, sid, (c[0] - 14, c[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    for poly in extra_lines:
        cv2.polylines(out, [np.int32(poly)], False, (255, 0, 255), 2)
    cv2.imwrite(path, out)
