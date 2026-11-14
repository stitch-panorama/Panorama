import cv2
import numpy as np
from stitcher.homography import ransac_homography

a = cv2.imread("data/cabinet/a.jpg")
b = cv2.imread("data/cabinet/b.jpg")
a = cv2.resize(a, (1000, int(a.shape[0] * 1000 / a.shape[1])))
b = cv2.resize(b, (1000, int(b.shape[0] * 1000 / b.shape[1])))

sift = cv2.SIFT_create()
kp_a, des_a = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
kp_b, des_b = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
pairs = cv2.BFMatcher().knnMatch(des_a, des_b, k=2)

print("ratio matches inliers iters cv2_inliers")
for ratio in [0.7, 0.8, 0.9]:
    good = [m for m, n in pairs if m.distance < ratio * n.distance]
    src = np.array([kp_a[m.queryIdx].pt for m in good])
    dst = np.array([kp_b[m.trainIdx].pt for m in good])
    H, mask, iters = ransac_homography(src, dst, thresh=3.0)
    _, m_cv = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    n_mine = 0 if mask is None else int(mask.sum())
    n_cv = 0 if m_cv is None else int(m_cv.sum())
    print(ratio, len(good), n_mine, iters, n_cv)