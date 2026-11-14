import numpy as np
from stitcher.homography import estimate_homography, apply_homography


def test_four_points_recover_exact_homography():
    H_true = np.array([[0.98, -0.05, 15],
                       [0.04, 0.98, -10],
                       [2e-5, -1e-5, 1]])
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 700, (4, 2))
    dst = apply_homography(H_true, src)

    H = estimate_homography(src, dst)

    assert np.abs(apply_homography(H, src) - dst).max() < 1e-8