import time
from dataclasses import dataclass

import cv2
import numpy as np

from stitcher.features import detect_and_describe, match_ratio_test
from stitcher.homography import ransac_homography
from stitcher.chain import chain_homographies
from stitcher.warp import canvas_bounds_n, warp_all
from stitcher.blend import compensate_chain, feather_blend_n


@dataclass
class StitchResult:
    panorama: np.ndarray | None   # BGR uint8, or None on failure
    ok: bool
    message: str                  # reason shown to the user when ok is False
    stats: dict


def _shrink(img, max_width):
    h, w = img.shape[:2]
    if max_width and w > max_width:
        return cv2.resize(img, (max_width, int(round(h * max_width / w))),
                          interpolation=cv2.INTER_AREA)
    return img


def is_stitchable(n_matches, n_inliers):
    """Brown & Lowe's rule of thumb."""
    return n_inliers > 8 + 0.3 * n_matches


def align_images(images, ratio=0.7, thresh=3.0):
    """Returns (ref_to, ref, pair_stats, error_message). ref_to[k] maps points of the
    reference image to image k. images must be in left-to-right order."""
    feats = [detect_and_describe(im) for im in images]
    pairwise, pair_stats = [], []
    for i in range(len(images) - 1):
        pts_a, des_a = feats[i]
        pts_b, des_b = feats[i + 1]
        idx = match_ratio_test(des_a, des_b, ratio)
        st = {"pair": (i + 1, i + 2), "matches": len(idx), "inliers": 0}
        pair_stats.append(st)
        label = f"Images {i + 1} and {i + 2}"
        if len(idx) < 4:
            return None, None, pair_stats, f"{label}: too few matching features."
        H, mask, _ = ransac_homography(pts_a[idx[:, 0]], pts_b[idx[:, 1]], thresh=thresh)
        if H is None:
            return None, None, pair_stats, f"{label}: no consistent alignment found."
        st["inliers"] = int(mask.sum())
        if not is_stitchable(len(idx), st["inliers"]):
            return None, None, pair_stats, (
                f"{label} do not appear to overlap enough, or the scene is too hard "
                "(glare, nearby objects, repetitive texture).")
        pairwise.append(H)
    ref = (len(images) - 1) // 2
    return chain_homographies(pairwise, ref), ref, pair_stats, ""


def stitch(images, ratio=0.7, thresh=3.0, max_width=1200):
    t0 = time.time()
    if len(images) < 2:
        return StitchResult(None, False, "Need at least two images.", {})
    images = [_shrink(im, max_width) for im in images]

    ref_to, ref, pair_stats, err = align_images(images, ratio, thresh)
    stats = {"pairs": pair_stats, "reference": ref}
    if ref_to is None:
        return StitchResult(None, False, err, stats)

    bounds = canvas_bounds_n([im.shape for im in images], ref_to)
    if bounds is None or bounds[2] * bounds[3] > 40_000_000:
        return StitchResult(None, False, "Alignment looks implausible (canvas too large).", stats)

    layers = warp_all(images, ref_to)
    canvases = [c for c, _ in layers]
    masks = [m for _, m in layers]
    pano = feather_blend_n(compensate_chain(canvases, masks, ref), masks)
    stats["canvas"] = (bounds[2], bounds[3])
    stats["runtime_s"] = round(time.time() - t0, 2)
    return StitchResult(pano, True, "ok", stats)