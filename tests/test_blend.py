import numpy as np
from stitcher.blend import feather_weights, feather_blend, estimate_gain, compensate_chain, feather_blend_n


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

def test_compensate_chain_matches_exposure():
    base = np.full((12, 12, 3), 100.0, np.float32)
    masks = [np.ones((12, 12), bool)] * 3
    out = compensate_chain([base * 0.64, base * 0.8, base], masks, ref=2)
    for o in out:
        assert np.allclose(o, 100.0, atol=1e-3)


def test_feather_blend_n_single_coverage_passes_through():
    canv = [np.full((4, 4, 3), v, np.float32) for v in (50, 100, 150)]
    masks = [np.zeros((4, 4), bool) for _ in range(3)]
    masks[0][:, :2] = True
    masks[1][:, 2:] = True
    out = feather_blend_n(canv, masks)
    assert (out[:, :2] == 50).all() and (out[:, 2:] == 100).all()