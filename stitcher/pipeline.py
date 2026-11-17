import time
from dataclasses import dataclass

import cv2
import numpy as np

from stitcher.features import detect_and_describe, match_ratio_test
from stitcher.homography import ransac_homography
from stitcher.warp import canvas_bounds, warp_to_canvas
from stitcher.blend import estimate_gain, feather_blend


@dataclass
class StitchResult:
    panorama: np.ndarray | None   # BGR uint8, or None on failure
    ok: bool
    message: str                  # reason shown to the user when ok is False
    stats: dict                   # matches, inliers, runtime, etc.


def _shrink(img, max_width):
    h, w = img.shape[:2]
    if max_width and w > max_width:
        scale = max_width / w
        return cv2.resize(img, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return img


def is_stitchable(n_matches, n_inliers):
    """Brown & Lowe's rule of thumb: enough inliers compared with the matches."""
    return n_inliers > 8 + 0.3 * n_matches


def stitch_pair(img_a, img_b, ratio=0.7, thresh=3.0, max_width=1200):
    t0 = time.time()
    img_a, img_b = _shrink(img_a, max_width), _shrink(img_b, max_width)

    pts_a, des_a = detect_and_describe(img_a)
    pts_b, des_b = detect_and_describe(img_b)
    idx = match_ratio_test(des_a, des_b, ratio)
    stats = {"keypoints_a": len(pts_a), "keypoints_b": len(pts_b),
             "matches": len(idx), "inliers": 0}

    if len(idx) < 4:
        return StitchResult(None, False, "Too few matching features between the two images.", stats)

    src, dst = pts_a[idx[:, 0]], pts_b[idx[:, 1]]
    H, mask, iters = ransac_homography(src, dst, thresh=thresh)
    if H is None:
        return StitchResult(None, False, "Could not find a consistent alignment.", stats)

    stats["inliers"], stats["iterations"] = int(mask.sum()), iters
    if not is_stitchable(len(idx), stats["inliers"]):
        return StitchResult(None, False,
                            "These images do not appear to overlap enough, or the scene is "
                            "too hard (glare, nearby objects, repetitive texture).", stats)

    _, _, W, Hc = canvas_bounds(img_a.shape, img_b.shape, H)
    if W * Hc > 40_000_000:
        return StitchResult(None, False, "Alignment looks implausible (canvas too large).", stats)

    ca, ma, cb, mb = warp_to_canvas(img_a, img_b, H)
    g = estimate_gain(ca, ma, cb, mb)
    pano = feather_blend(ca, ma, np.clip(cb * g, 0, 255), mb)
    stats["runtime_s"] = round(time.time() - t0, 2)
    return StitchResult(pano, True, "ok", stats)


def stitch(images, ratio=0.7, thresh=3.0, max_width=1200):
    if len(images) < 2:
        return StitchResult(None, False, "Need at least two images.", {})
    if len(images) == 2:
        return stitch_pair(images[0], images[1], ratio, thresh, max_width)
    return StitchResult(None, False, "More than two images is not supported yet.", {})