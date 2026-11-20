import argparse
import csv
import os
import time

import cv2
import numpy as np

from stitcher.features import detect_and_describe, match_ratio_test
from stitcher.homography import apply_homography, estimate_homography, ransac_homography

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/hpatches/hpatches-sequences-release")
ap.add_argument("--ratio", type=float, default=0.7)
ap.add_argument("--thresh", type=float, default=3.0)
ap.add_argument("--max-width", type=int, default=1200)
ap.add_argument("--every", type=int, default=1,
                help="use every N-th sequence (e.g. 12 for a quick test mixing i_ and v_)")
ap.add_argument("--out", default="results/hpatches_baseline.csv")
args = ap.parse_args()

METHODS = ["ours", "opencv", "lstsq"]


def load(path):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"could not read {path}")
    h, w = img.shape[:2]
    S = np.eye(3)
    if w > args.max_width:
        nw = args.max_width
        nh = int(round(h * nw / w))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        S = np.diag([nw / w, nh / h, 1.0])
    return img, S


def corner_error(H_est, H_gt, w, h):
    """Mean distance (px) between where the two homographies send the image corners."""
    if H_est is None:
        return np.inf
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
    with np.errstate(all="ignore"):
        d = np.linalg.norm(apply_homography(H_est, corners) -
                           apply_homography(H_gt, corners), axis=1).mean()
    return float(d) if np.isfinite(d) else np.inf


seqs = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
seqs = seqs[::args.every]

rows = []
for si, seq in enumerate(seqs):
    d = os.path.join(args.root, seq)
    img1, S1 = load(os.path.join(d, "1.ppm"))
    h, w = img1.shape[:2]
    pts1, des1 = detect_and_describe(img1)

    for k in range(2, 7):
        imgk, Sk = load(os.path.join(d, f"{k}.ppm"))
        H_gt = Sk @ np.loadtxt(os.path.join(d, f"H_1_{k}")) @ np.linalg.inv(S1)
        ptsk, desk = detect_and_describe(imgk)
        idx = match_ratio_test(des1, desk, args.ratio)
        n = len(idx)

        res = {m: (None, 0, 0.0) for m in METHODS}
        if n >= 4:
            src, dst = pts1[idx[:, 0]], ptsk[idx[:, 1]]

            t = time.perf_counter()
            H, mask, _ = ransac_homography(src, dst, thresh=args.thresh)
            res["ours"] = (H, 0 if mask is None else int(mask.sum()), time.perf_counter() - t)

            t = time.perf_counter()
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, args.thresh)
            res["opencv"] = (H, 0 if mask is None else int(mask.sum()), time.perf_counter() - t)

            t = time.perf_counter()
            try:
                H = estimate_homography(src, dst)
            except np.linalg.LinAlgError:
                H = None
            res["lstsq"] = (H, n, time.perf_counter() - t)

        for m in METHODS:
            H, inl, dt = res[m]
            rows.append({"sequence": seq, "kind": seq[0], "pair": k, "matches": n,
                         "method": m, "inliers": inl,
                         "error_px": corner_error(H, H_gt, w, h), "time_s": round(dt, 4)})
    print(f"[{si + 1}/{len(seqs)}] {seq}", flush=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)

print("\nkind method pairs  acc@1  acc@3  acc@5  failed  est_ms")
for kind in ["i", "v", "all"]:
    for m in METHODS:
        sel = [r for r in rows if r["method"] == m and (kind == "all" or r["kind"] == kind)]
        e = np.array([r["error_px"] for r in sel])
        print(f"{kind:>4} {m:>6} {len(sel):5d} {np.mean(e <= 1):6.2f} {np.mean(e <= 3):6.2f} "
              f"{np.mean(e <= 5):6.2f} {int(np.isinf(e).sum()):6d} "
              f"{1000 * np.mean([r['time_s'] for r in sel]):7.1f}")
print("saved", args.out)