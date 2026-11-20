import argparse
import os
import glob
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("source", help="path to a sharp, wide photo")
ap.add_argument("outdir", help="folder for the views and the ground truth")
ap.add_argument("-n", type=int, default=5, help="number of views")
ap.add_argument("--view-frac", type=float, default=0.4,
                help="width of each view as a fraction of the photo")
ap.add_argument("--margin", type=int, default=60,
                help="pixels cropped from every side to remove warp borders (must be > 0)")
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
img = cv2.imread(args.source)
if img is None:
    raise SystemExit(f"could not read {args.source}")
img = cv2.resize(img, (1200, int(img.shape[0] * 1200 / img.shape[1])))
h, w = img.shape[:2]

n, m = args.n, args.margin
cw = int(args.view_frac * w)
step = (w - cw) // (n - 1)
x0 = [k * step for k in range(n)]
ref = (n - 1) // 2                       # same rule the pipeline uses


def shift(dx, dy=0):
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], float)


def small_h():
    a = np.deg2rad(rng.uniform(-3, 3))
    s = rng.uniform(0.97, 1.03)
    tx, ty = rng.uniform(-10, 10, 2)
    px, py = rng.uniform(-2e-5, 2e-5, 2)
    return np.array([[s * np.cos(a), -s * np.sin(a), tx],
                     [s * np.sin(a),  s * np.cos(a), ty],
                     [px, py, 1.0]])


os.makedirs(args.outdir, exist_ok=True)
for f in glob.glob(os.path.join(args.outdir, "view_*.jpg")):
    os.remove(f)
truth = []
for k in range(n):
    Hs = np.eye(3) if k == ref else small_h()
    warped = cv2.warpPerspective(img[:, x0[k]:x0[k] + cw], Hs, (cw, h))
    view = warped[m:h - m, m:cw - m]                     # drop the black borders
    cv2.imwrite(os.path.join(args.outdir, f"view_{k}.jpg"), view)
    # reference view pixel -> uncropped reference crop -> source photo -> crop k
    # -> warped view k -> cropped view k   (the rightmost matrix is applied first)
    truth.append(shift(-m, -m) @ Hs @ shift(-x0[k]) @ shift(x0[ref]) @ shift(m, m))
np.save(os.path.join(args.outdir, "ref_to_true.npy"), np.array(truth))
print("views:", n, "reference:", ref, "overlap between neighbors:", cw - 2 * m - step, "px")