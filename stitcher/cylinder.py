import cv2
import numpy as np


def _best(eqs, scale):
    """Of the (numerator, denominator) pairs, use the best-conditioned one."""
    num, den = max(eqs, key=lambda e: abs(e[1]))
    if abs(den) <= 1e-3 * scale:
        return None
    f2 = num / den
    return float(np.sqrt(f2)) if f2 > 0 else None


def estimate_focal(H, w, h):
    """Focal length candidates (pixels) from a homography H (raw pixel coordinates) between
    two images from a camera that only rotates: K^-1 H K must be a rotation."""
    C = np.array([[1, 0, -(w - 1) / 2], [0, 1, -(h - 1) / 2], [0, 0, 1]], float)
    M = C @ H @ np.linalg.inv(C)            # H in coordinates centred on the image
    M = M / M[2, 2]
    (h11, h12, h13), (h21, h22, h23), (h31, h32, _) = M
    rows = _best([(-h13 * h23, h11 * h21 + h12 * h22),
                  (h23 ** 2 - h13 ** 2, h11 ** 2 + h12 ** 2 - h21 ** 2 - h22 ** 2)],
                 h11 ** 2 + h12 ** 2 + h21 ** 2 + h22 ** 2)
    cols = _best([(-(h11 * h12 + h21 * h22), h31 * h32),
                  (h11 ** 2 + h21 ** 2 - h12 ** 2 - h22 ** 2, h32 ** 2 - h31 ** 2)],
                 h31 ** 2 + h32 ** 2)
    return [f for f in (rows, cols) if f is not None]


def estimate_focal_from_pairs(Hs, w, h):
    """Median focal length over all pairs, or None if nothing plausible was found."""
    lo, hi = 0.25 * max(w, h), 6 * max(w, h)
    cands = [f for H in Hs for f in estimate_focal(H, w, h) if lo < f < hi]
    return float(np.median(cands)) if cands else None


def cylinder_maps(w, h, f):
    """Lookup maps that resample a w x h image onto a cylinder of radius f (pixels).
    The output is cropped to rows and columns that stay inside the source image."""
    theta_max = np.arctan(((w - 1) / 2) / f)
    out_w = int(2 * f * theta_max)
    out_h = int(np.floor((h - 1) * np.cos(theta_max)))
    u = np.arange(out_w) - (out_w - 1) / 2
    v = np.arange(out_h) - (out_h - 1) / 2
    uu, vv = np.meshgrid(u, v)
    theta = uu / f
    map_x = f * np.tan(theta) + (w - 1) / 2
    map_y = vv / np.cos(theta) + (h - 1) / 2
    return map_x.astype(np.float32), map_y.astype(np.float32)


def cylindrical_warp(img, f):
    h, w = img.shape[:2]
    map_x, map_y = cylinder_maps(w, h, f)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)