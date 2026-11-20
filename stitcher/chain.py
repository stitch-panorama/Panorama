import numpy as np


def chain_homographies(pairwise, ref):
    """pairwise[i] maps points of image i to image i+1 (3x3).
    Returns ref_to, where ref_to[k] maps points of the reference image to image k."""
    n = len(pairwise) + 1
    ref_to = [None] * n
    ref_to[ref] = np.eye(3)
    for k in range(ref + 1, n):            # to the right: earlier chain is applied first
        ref_to[k] = pairwise[k - 1] @ ref_to[k - 1]
    for k in range(ref - 1, -1, -1):       # to the left: step back with inverses
        ref_to[k] = np.linalg.inv(pairwise[k]) @ ref_to[k + 1]
    return [H / H[2, 2] for H in ref_to]