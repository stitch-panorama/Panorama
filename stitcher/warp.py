import numpy as np
from stitcher.homography import apply_homography


def canvas_bounds(shape_ref, shape_other, H_ref_to_other):
    """Bounding box of both images in the reference frame.
    Returns (xmin, ymin, width, height) of the canvas."""
    h1, w1 = shape_ref[:2]
    h2, w2 = shape_other[:2]
    corners_ref = np.array([[0, 0], [w1 - 1, 0], [w1 - 1, h1 - 1], [0, h1 - 1]], float)
    corners_other = np.array([[0, 0], [w2 - 1, 0], [w2 - 1, h2 - 1], [0, h2 - 1]], float)
    # the other image's corners, expressed in the reference frame
    corners_other_in_ref = apply_homography(np.linalg.inv(H_ref_to_other), corners_other)
    allc = np.vstack([corners_ref, corners_other_in_ref])
    xmin, ymin = np.floor(allc.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(allc.max(axis=0)).astype(int)
    return int(xmin), int(ymin), int(xmax - xmin + 1), int(ymax - ymin + 1)


def bilinear_sample(img, xs, ys):
    """Sample a color image at float coordinates. Returns (values, valid_mask)."""
    h, w = img.shape[:2]
    valid = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)   # False for NaN too
    x = np.where(valid, xs, 0.0)
    y = np.where(valid, ys, 0.0)
    x0 = np.minimum(np.floor(x).astype(int), w - 2)
    y0 = np.minimum(np.floor(y).astype(int), h - 2)
    wx = (x - x0)[..., None]
    wy = (y - y0)[..., None]
    img = img.astype(np.float32)
    top = (1 - wx) * img[y0, x0] + wx * img[y0, x0 + 1]
    bot = (1 - wx) * img[y0 + 1, x0] + wx * img[y0 + 1, x0 + 1]
    return (1 - wy) * top + wy * bot, valid


def warp_to_canvas(ref, other, H_ref_to_other):
    """Place both images on one canvas in ref's frame.
    H_ref_to_other maps points in ref to points in other.
    Returns (canvas_ref, mask_ref, canvas_other, mask_other)."""
    xmin, ymin, W, Hc = canvas_bounds(ref.shape, other.shape, H_ref_to_other)

    # every canvas pixel (u, v) -> coordinates in the reference frame
    u, v = np.meshgrid(np.arange(W), np.arange(Hc))
    xr = (u + xmin).astype(float)
    yr = (v + ymin).astype(float)

    # reference image: pure translation, no homography needed
    canvas_ref, mask_ref = bilinear_sample(ref, xr, yr)

    # other image: reference-frame coordinates -> other's coordinates via H
    pts = np.stack([xr.ravel(), yr.ravel(), np.ones(xr.size)])      # 3 x N
    with np.errstate(all="ignore"):
        q = H_ref_to_other @ pts
        xo = (q[0] / q[2]).reshape(xr.shape)
        yo = (q[1] / q[2]).reshape(xr.shape)
    canvas_other, mask_other = bilinear_sample(other, xo, yo)
    return canvas_ref, mask_ref, canvas_other, mask_other


def composite_average(c1, m1, c2, m2):
    """Simple composite: average where both images exist. (Real blending comes later.)"""
    out = np.zeros_like(c1)
    both = m1 & m2
    out[m1 & ~m2] = c1[m1 & ~m2]
    out[m2 & ~m1] = c2[m2 & ~m1]
    out[both] = 0.5 * c1[both] + 0.5 * c2[both]
    return out.astype(np.uint8)