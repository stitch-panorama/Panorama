import argparse
import glob
import os
import sys

import cv2
import numpy as np

from stitcher.homography import apply_homography
from stitcher.pipeline import align_images, stitch

ap = argparse.ArgumentParser()
ap.add_argument("folder", nargs="?", default="data/synthetic/views")
args = ap.parse_args()

truth = np.load(os.path.join(args.folder, "ref_to_true.npy"))
n = len(truth)
views = [cv2.imread(os.path.join(args.folder, f"view_{k}.jpg")) for k in range(n)]

ref_to, ref, pair_stats, err = align_images(views)
if ref_to is None:
    sys.exit("alignment failed: " + err)

h, w = views[ref].shape[:2]
corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)

for st in pair_stats:
    print("pair", st["pair"], "matches", st["matches"], "inliers", st["inliers"])
print("reference image:", ref)
print("image steps_from_ref corner_error_px")
for k in range(n):
    diff = apply_homography(ref_to[k], corners) - apply_homography(truth[k], corners)
    print(k, abs(k - ref), round(float(np.linalg.norm(diff, axis=1).mean()), 2))

res = stitch(views)
print(res.ok, res.message, {k: v for k, v in res.stats.items() if k != "pairs"})
name = os.path.basename(os.path.normpath(args.folder))
if res.ok:
    cv2.imwrite(f"results/stitch_{name}_{n}views.jpg", res.panorama)