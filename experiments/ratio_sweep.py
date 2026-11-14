import cv2
import numpy as np

c = cv2.imread("data/synthetic/c.jpg")
d = cv2.imread("data/synthetic/d.jpg")
H_true = np.load("data/synthetic/H_true.npy")

sift = cv2.SIFT_create()
kp_c, des_c = sift.detectAndCompute(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), None)
kp_d, des_d = sift.detectAndCompute(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY), None)

# for each descriptor in c, get its 2 nearest neighbors in d
pairs = cv2.BFMatcher().knnMatch(des_c, des_d, k=2)

for ratio in [0.6, 0.7, 0.8, 0.9]:
    # Lowe's ratio test
    good = [m for m, n in pairs if m.distance < ratio * n.distance]

    # ground truth: where should this point land in d?
    correct = 0
    for m in good:
        p = np.array([*kp_c[m.queryIdx].pt, 1.0])
        q = H_true @ p
        q = q[:2] / q[2]
        if np.linalg.norm(q - np.array(kp_d[m.trainIdx].pt)) < 3:
            correct += 1

    print(ratio, len(kp_c), len(kp_d), len(pairs), len(good), correct)

    if ratio == 0.7:
        out = cv2.drawMatches(c, kp_c, d, kp_d, good, None)
        cv2.imwrite("results/matches_synthetic.jpg", out)