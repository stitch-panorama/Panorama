import numpy as np
from stitcher.chain import chain_homographies
from stitcher.homography import apply_homography


def shift(dx):
    return np.array([[1, 0, dx], [0, 1, 0], [0, 0, 1]], float)


def test_chain_with_shifts():
    ref_to = chain_homographies([shift(-100), shift(-80), shift(-60)], ref=1)
    p = np.array([[500.0, 200.0]])
    assert np.allclose(apply_homography(ref_to[1], p), [[500, 200]])
    assert np.allclose(apply_homography(ref_to[2], p), [[420, 200]])
    assert np.allclose(apply_homography(ref_to[3], p), [[360, 200]])
    assert np.allclose(apply_homography(ref_to[0], p), [[600, 200]])


def test_chain_order_matters():
    scale2 = np.diag([2.0, 2.0, 1.0])
    ref_to = chain_homographies([shift(10), scale2], ref=0)
    # x = 5 -> shifted to 15 -> scaled to 30 (the wrong order would give 20)
    assert np.allclose(apply_homography(ref_to[2], np.array([[5.0, 0.0]])), [[30, 0]])