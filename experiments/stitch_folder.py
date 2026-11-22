import argparse
import glob
import os
import time

import cv2

from stitcher.pipeline import stitch

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--reverse", action="store_true", help="reverse the image order")
ap.add_argument("--ratio", type=float, default=0.7)
ap.add_argument("--thresh", type=float, default=3.0)
ap.add_argument("--projection", choices=["auto", "planar", "cylindrical"], default="auto")
ap.add_argument("--focal-ratio", type=float, default=None,
                help="focal length as a multiple of image width (estimated if omitted)")
args = ap.parse_args()

paths = sorted(p for e in ("*.jpg", "*.jpeg", "*.png") for p in glob.glob(os.path.join(args.folder, e)))
if args.reverse:
    paths = paths[::-1]
images = [cv2.imread(p) for p in paths]
name = os.path.basename(os.path.normpath(args.folder))
print(len(images), "images:", [os.path.basename(p) for p in paths])

res = stitch(images, ratio=args.ratio, thresh=args.thresh,
             projection=args.projection, focal_ratio=args.focal_ratio)
print("ours:", res.ok, res.message, {k: v for k, v in res.stats.items() if k != "pairs"})
for p in res.stats.get("pairs", []):
    print("  pair", p["pair"], "matches", p["matches"], "inliers", p["inliers"])
if res.ok:
    cv2.imwrite(f"results/real_{name}_ours.jpg", res.panorama)

# baseline: OpenCV's high-level stitcher (it works out the image order by itself)
t = time.time()
status, pano = cv2.Stitcher_create(cv2.Stitcher_PANORAMA).stitch(images)
print("OpenCV Stitcher status", status, "(0 = OK),", round(time.time() - t, 2), "s")
if status == 0:
    cv2.imwrite(f"results/real_{name}_opencv.jpg", pano)