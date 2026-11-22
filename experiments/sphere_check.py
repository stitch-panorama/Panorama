import argparse
import os
import sys

import cv2
import numpy as np

from stitcher.cylinder import estimate_focal, estimate_focal_from_pairs
from stitcher.homography import apply_homography
from stitcher.pipeline import align_images, stitch

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--projection", choices=["auto", "planar", "cylindrical"], default="auto")
args = ap.parse_args()

truth = np.load(os.path.join(args.folder, "truth.npz"))
f_true, K, Rs = float(truth["f"]), truth["K"], truth["Rs"]
n = len(Rs)
views = [cv2.imread(os.path.join(args.folder, f"view_{k}.jpg")) for k in range(n)]
h, w = views[0].shape[:2]

ref_to, ref, pair_stats, err = align_images(views)
if err:
    sys.exit("alignment failed: " + err)

corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
print("pair  matches  inliers  corner_error_px (against the true homography)")
for i, p in enumerate(pair_stats):
    H_true = K @ Rs[i + 1].T @ Rs[i] @ np.linalg.inv(K)
    d = np.linalg.norm(apply_homography(p["H"], corners) - apply_homography(H_true, corners), axis=1).mean()
    print(p["pair"], p["matches"], p["inliers"], round(float(d), 2))

Hs = [p["H"] for p in pair_stats]
f_est = estimate_focal_from_pairs(Hs, w, h)
print("true focal length (px):", round(f_true, 1))
print("candidates per pair:", [[round(c, 1) for c in estimate_focal(H, w, h)] for H in Hs])
if f_est:
    print("estimate (median):", round(f_est, 1), " error:", f"{100 * (f_est - f_true) / f_true:+.1f} %")

res = stitch(views, projection=args.projection)
print(res.ok, res.message, {k: v for k, v in res.stats.items() if k != "pairs"})
if res.ok:
    name = os.path.basename(os.path.normpath(args.folder))
    cv2.imwrite(f"results/sphere_{name}_{args.projection}.jpg", res.panorama)