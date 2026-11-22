import argparse
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("source", help="equirectangular 360 x 180 panorama (width = 2 x height)")
ap.add_argument("outdir")
ap.add_argument("-n", type=int, default=6, help="number of views")
ap.add_argument("--step", type=float, default=30.0, help="yaw step between views (degrees)")
ap.add_argument("--hfov", type=float, default=60.0, help="horizontal field of view of each view (degrees)")
ap.add_argument("--yaw0", type=float, default=0.0, help="yaw of the first view (degrees)")
ap.add_argument("--width", type=int, default=1000)
ap.add_argument("--height", type=int, default=750)
args = ap.parse_args()

img = cv2.imread(args.source)
if img is None:
    raise SystemExit(f"could not read {args.source}")
He, We = img.shape[:2]
if abs(We / He - 2) > 0.05:
    raise SystemExit("this is not a 2:1 equirectangular image")

w, h = args.width, args.height
f = (w / 2) / np.tan(np.deg2rad(args.hfov) / 2)
cx, cy = (w - 1) / 2, (h - 1) / 2
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

uu, vv = np.meshgrid(np.arange(w), np.arange(h))
rays = np.stack([uu - cx, vv - cy, np.full((h, w), f)], axis=-1)   # camera coordinates


def yaw_matrix(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


os.makedirs(args.outdir, exist_ok=True)
Rs = []
for k in range(args.n):
    R = yaw_matrix(np.deg2rad(args.yaw0 + k * args.step))
    Rs.append(R)
    d = rays @ R.T                                   # the same rays in world coordinates
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arctan2(-d[..., 1], np.hypot(d[..., 0], d[..., 2]))
    map_x = ((lon / (2 * np.pi) + 0.5) * We - 0.5).astype(np.float32)
    map_y = ((0.5 - lat / np.pi) * He - 0.5).astype(np.float32)
    view = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    cv2.imwrite(os.path.join(args.outdir, f"view_{k}.jpg"), view, [cv2.IMWRITE_JPEG_QUALITY, 95])

np.savez(os.path.join(args.outdir, "truth.npz"), f=f, K=K, Rs=np.array(Rs))
print(f"views: {args.n}, true focal length: {f:.1f} px, "
      f"overlap: {100 * (1 - args.step / args.hfov):.0f} %, "
      f"total span: {(args.n - 1) * args.step + args.hfov:.0f} degrees")