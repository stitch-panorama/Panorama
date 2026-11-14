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

def estimate_gain(c1, m1, c2, m2):
    """Per-channel gain g so that c2 * g matches c1 on their overlap (mean brightness)."""
    both = m1 & m2
    if both.sum() < 100:
        return np.ones(3)
    return c1[both].mean(axis=0) / np.maximum(c2[both].mean(axis=0), 1e-6)