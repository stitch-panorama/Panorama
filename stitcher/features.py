import cv2
import numpy as np


def detect_and_describe(img):
    """SIFT keypoint positions (N, 2) and descriptors (N, 128) for a BGR image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps, des = cv2.SIFT_create().detectAndCompute(gray, None)
    pts = np.array([k.pt for k in kps], dtype=float).reshape(-1, 2)
    return pts, des


def match_ratio_test(des_a, des_b, ratio=0.7):
    """Index pairs (i, j): descriptor i in image A matches descriptor j in image B."""
    if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
        return np.empty((0, 2), dtype=int)
    pairs = cv2.BFMatcher().knnMatch(des_a, des_b, k=2)
    good = [(m.queryIdx, m.trainIdx) for m, n in pairs if m.distance < ratio * n.distance]
    return np.array(good, dtype=int).reshape(-1, 2)