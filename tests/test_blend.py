import numpy as np
from stitcher.blend import feather_weights, feather_blend, estimate_gain


def test_weights_zero_outside_and_peak_in_middle():
    m = np.zeros((9, 9), bool)
    m[2:7, 2:7] = True
    w = feather_weights(m)
    assert w[0, 0] == 0
    assert w[4, 4] == w.max()
    assert w[2, 2] < w[4, 4]


def test_single_image_passes_through():
    c1 = np.full((4, 4, 3), 100, np.uint8)
    c2 = np.full((4, 4, 3), 200, np.uint8)
    m1 = np.ones((4, 4), bool)
    m2 = np.zeros((4, 4), bool)
    assert (feather_blend(c1, m1, c2, m2) == 100).all()

def test_gain_recovers_known_exposure_change():
    rng = np.random.default_rng(0)
    c1 = rng.uniform(50, 200, (20, 20, 3)).astype(np.float32)
    c2 = c1 * 0.5
    m = np.ones((20, 20), bool)
    assert np.allclose(estimate_gain(c1, m, c2, m), 2.0, atol=1e-4)