import numpy as np

from stitcher.cylinder import cylindrical_warp, estimate_focal, estimate_focal_from_pairs


def rot(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def homography_for(f, R, w, h):
    K = np.diag([f, f, 1.0])
    C = np.array([[1, 0, -(w - 1) / 2], [0, 1, -(h - 1) / 2], [0, 0, 1]])
    return np.linalg.inv(C) @ K @ R @ np.linalg.inv(K) @ C


def test_focal_recovered_from_rotation_homography():
    f, w, h = 800.0, 1000, 750
    estimates = estimate_focal(homography_for(f, rot(0.35, 0.08, 0.05), w, h), w, h)
    assert len(estimates) >= 1
    assert all(abs(e - f) < 1e-3 for e in estimates)


def test_focal_from_pairs():
    f, w, h = 900.0, 1000, 750
    Hs = [homography_for(f, rot(y, 0.03, 0.02), w, h) for y in (0.3, 0.4, 0.5)]
    assert abs(estimate_focal_from_pairs(Hs, w, h) - f) < 1e-3


def test_cylinder_crop_has_no_empty_pixels():
    img = np.full((60, 100, 3), 200, np.uint8)
    out = cylindrical_warp(img, 80.0)
    assert out.shape[1] < img.shape[1] and out.shape[0] <= img.shape[0]
    assert out.min() >= 190