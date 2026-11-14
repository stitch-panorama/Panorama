import cv2
import numpy as np
from stitcher.homography import estimate_homography, apply_homography

c = cv2.imread("data/synthetic/c.jpg")
d = cv2.imread("data/synthetic/d.jpg")
H_true = np.load("data/synthetic/H_true.npy")

sift = cv2.SIFT_create()
kp_c, des_c = sift.detectAndCompute(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), None)
kp_d, des_d = sift.detectAndCompute(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY), None)
pairs = cv2.BFMatcher().knnMatch(des_c, des_d, k=2)

h, w = c.shape[:2]
corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)

def corner_error(H):
    # average pixel distance between where H and H_true send the image corners
    diff = apply_homography(H, corners) - apply_homography(H_true, corners)
    return np.linalg.norm(diff, axis=1).mean()

for ratio in [0.6, 0.7, 0.8, 0.9]:
    good = [m for m, n in pairs if m.distance < ratio * n.distance]
    src = np.array([kp_c[m.queryIdx].pt for m in good])
    dst = np.array([kp_d[m.trainIdx].pt for m in good])
    ok = np.linalg.norm(apply_homography(H_true, src) - dst, axis=1) < 3
    err_all = corner_error(estimate_homography(src, dst))
    err_ok = corner_error(estimate_homography(src[ok], dst[ok]))
    print(ratio, len(good), (~ok).sum(), round(err_all, 2), round(err_ok, 2))