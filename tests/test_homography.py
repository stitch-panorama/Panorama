import numpy as np
from stitcher.homography import estimate_homography, apply_homography, ransac_homography


def test_four_points_recover_exact_homography():
    H_true = np.array([[0.98, -0.05, 15],
                       [0.04, 0.98, -10],
                       [2e-5, -1e-5, 1]])
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 700, (4, 2))
    dst = apply_homography(H_true, src)

    H = estimate_homography(src, dst)

    assert np.abs(apply_homography(H, src) - dst).max() < 1e-8

def test_ransac_rejects_outliers():
    H_true = np.array([[0.98, -0.05, 15], [0.04, 0.98, -10], [2e-5, -1e-5, 1]])
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 700, (200, 2))
    dst = apply_homography(H_true, src) + rng.normal(0, 0.3, (200, 2))
    bad = rng.choice(200, 60, replace=False)
    dst[bad] = rng.uniform(0, 700, (60, 2))

    H, mask, _ = ransac_homography(src, dst, thresh=3.0)

    assert mask[bad].sum() <= 2          # corrupted points rejected
    assert mask.sum() >= 135             # good points kept
    probe = rng.uniform(0, 700, (50, 2))
    assert np.abs(apply_homography(H, probe) - apply_homography(H_true, probe)).max() < 1.0