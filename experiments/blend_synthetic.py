import cv2
import numpy as np
from stitcher.homography import ransac_homography
from stitcher.warp import canvas_bounds, warp_to_canvas, composite_average
from stitcher.blend import feather_blend

c = cv2.imread("data/synthetic/c.jpg")
d = cv2.imread("data/synthetic/d.jpg")

# same matching + RANSAC as before (H maps c -> d)
sift = cv2.SIFT_create()
kp_c, des_c = sift.detectAndCompute(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), None)
kp_d, des_d = sift.detectAndCompute(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY), None)
pairs = cv2.BFMatcher().knnMatch(des_c, des_d, k=2)
good = [m for m, n in pairs if m.distance < 0.7 * n.distance]
src = np.array([kp_c[m.queryIdx].pt for m in good])
dst = np.array([kp_d[m.trainIdx].pt for m in good])
H, mask, iters = ransac_homography(src, dst, thresh=3.0)

rc, mc, rd, md = warp_to_canvas(c, d, H)

# synthetic-only: d has black borders from make_pair, so mark them invalid.
# Real photos have no such borders, so skip this step for them.
valid_d = (d.sum(axis=2) > 15).astype(np.uint8)
valid_d = cv2.erode(valid_d, np.ones((7, 7), np.uint8))
_, _, vd, _ = warp_to_canvas(c, valid_d.astype(np.float32)[..., None], H)
md_clean = md & (vd[..., 0] >= 0.99)

# ground truth: c's frame is the source photo's frame
source = cv2.imread("data/synthetic/source.jpg")
source = cv2.resize(source, (1200, int(source.shape[0] * 1200 / source.shape[1])))
xmin, ymin, W, Hc = canvas_bounds(c.shape, d.shape, H)

def error_vs_source(img, covered):
    y0, x0 = max(0, -ymin), max(0, -xmin)
    y1, x1 = min(Hc, source.shape[0] - ymin), min(W, source.shape[1] - xmin)
    a = img[y0:y1, x0:x1].astype(float)
    b = source[y0 + ymin:y1 + ymin, x0 + xmin:x1 + xmin].astype(float)
    return np.abs(a - b)[covered[y0:y1, x0:x1]].mean()

cover = mc | md_clean      # same set of pixels for every variant
avg_raw = composite_average(rc, mc, rd, md)
avg_clean = composite_average(rc, mc, rd, md_clean)
feather = feather_blend(rc, mc, rd, md_clean)

print("mean abs error vs source photo (0-255):")
print("  average, raw masks:         ", round(error_vs_source(avg_raw, cover), 2))
print("  average, border-free masks: ", round(error_vs_source(avg_clean, cover), 2))
print("  feather, border-free masks: ", round(error_vs_source(feather, cover), 2))
cv2.imwrite("results/blend_feather.jpg", feather)