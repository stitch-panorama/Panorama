import glob

import cv2

from stitcher.cylinder import estimate_focal
from stitcher.pipeline import _shrink, align_images
from stitcher.warp import canvas_bounds_n

paths = sorted(glob.glob("data/real/boat/*.jpg"))
images = [_shrink(cv2.imread(p), 1200) for p in paths]
ref_to, ref, pair_stats, err = align_images(images)
h, w = images[0].shape[:2]

print("focal length candidates per pair (pixels):")
for p in pair_stats:
    print(" ", p["pair"], [round(f, 1) for f in estimate_focal(p["H"], w, h)])
print("flat layout canvas (xmin, ymin, width, height):",
      canvas_bounds_n([im.shape for im in images], ref_to))