import glob
import time

import cv2

from stitcher.features import detect_and_describe

paths = sorted(glob.glob("data/hpatches/hpatches-sequences-release/v_*/1.ppm"))[:20]
imgs = [cv2.imread(p) for p in paths]
for det in ("sift", "orb"):
    t = time.perf_counter()
    n = sum(len(detect_and_describe(im, det)[0]) for im in imgs)
    ms = (time.perf_counter() - t) / len(imgs) * 1000
    print(det, round(ms, 1), "ms per image,", n // len(imgs), "keypoints per image")