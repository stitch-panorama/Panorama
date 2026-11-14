import cv2
import numpy as np


def feather_weights(mask):
    """Distance from each valid pixel to the nearest invalid pixel (0 outside the mask)."""
    m = mask.astype(np.uint8)
    # pad with a zero border so the canvas edge counts as an edge
    padded = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)
    return dist[1:-1, 1:-1]


def feather_blend(c1, m1, c2, m2):
    """Blend two canvases with distance-based weights."""
    w1 = feather_weights(m1)[..., None]
    w2 = feather_weights(m2)[..., None]
    out = (w1 * c1 + w2 * c2) / np.maximum(w1 + w2, 1e-6)
    return np.rint(np.clip(out, 0, 255)).astype(np.uint8)