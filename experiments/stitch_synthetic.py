import cv2
import numpy as np
from stitcher.homography import ransac_homography
from stitcher.warp import warp_to_canvas, composite_average

c = cv2.imread("data/synthetic/c.jpg")
d = cv2.imread("data/synthetic/d.jpg")

sift = cv2.SIFT_create()
kp_c, des_c = sift.detectAndCompute(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), None)
kp_d, des_d = sift.detectAndCompute(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY), None)
pairs = cv2.BFMatcher().knnMatch(des_c, des_d, k=2)
good = [m for m, n in pairs if m.distance < 0.7 * n.distance]
src = np.array([kp_c[m.queryIdx].pt for m in good])
dst = np.array([kp_d[m.trainIdx].pt for m in good])

H, mask, iters = ransac_homography(src, dst, thresh=3.0)   # maps c -> d

def overlap_error(H):
    rc, mc, rd, md = warp_to_canvas(c, d, H)
    both = mc & md
    return np.median(np.abs(rc[both] - rd[both]))

H_bad = H.copy()
H_bad[0, 2] += 5          # deliberately wrong by 5 pixels

print("overlap error, RANSAC H:", round(float(overlap_error(H)), 2))
print("overlap error, H shifted 5px:", round(float(overlap_error(H_bad)), 2))

rc, mc, rd, md = warp_to_canvas(c, d, H)
print("canvas size (w, h):", rc.shape[1], rc.shape[0])
cv2.imwrite("results/stitch_synthetic.jpg", composite_average(rc, mc, rd, md))