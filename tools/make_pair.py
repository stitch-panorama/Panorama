import cv2
import numpy as np

img = cv2.imread("data/synthetic/source.jpg")
img = cv2.resize(img, (1200, int(img.shape[0] * 1200 / img.shape[1])))
h, w = img.shape[:2]

x0 = int(0.35 * w)                 # left edge of the second view
a = img[:, :int(0.65 * w)]         # left 65%
b_raw = img[:, x0:]                # right 65%

# a known transformation applied to the second view
Hs = np.array([[0.98, -0.05, 15],
               [0.04,  0.98, -10],
               [2e-5, -1e-5,  1]])
b = cv2.warpPerspective(b_raw, Hs, (b_raw.shape[1], b_raw.shape[0]))

# the true homography mapping points in a to points in b
T = np.array([[1, 0, -x0],
              [0, 1, 0],
              [0, 0, 1]], dtype=float)
H_true = Hs @ T
H_true = H_true / H_true[2, 2]     # normalize so the bottom-right is 1
print(H_true)

cv2.imwrite("data/synthetic/c.jpg", a)
cv2.imwrite("data/synthetic/d.jpg", b)
np.save("data/synthetic/H_true.npy", H_true)