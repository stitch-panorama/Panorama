import numpy as np
from stitcher.warp import canvas_bounds


def test_identity_canvas_is_image_size():
    xmin, ymin, W, H = canvas_bounds((100, 200, 3), (100, 200, 3), np.eye(3))
    assert (xmin, ymin, W, H) == (0, 0, 200, 100)


def test_shifted_image_extends_canvas():
    # a point at x in the reference frame is at x - 50 in the other image,
    # so the other image sits 50 pixels to the right of the reference
    Hs = np.array([[1, 0, -50], [0, 1, 0], [0, 0, 1]], float)
    xmin, ymin, W, H = canvas_bounds((100, 200, 3), (100, 200, 3), Hs)
    assert (xmin, ymin, W, H) == (0, 0, 250, 100)