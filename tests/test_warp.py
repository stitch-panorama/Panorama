import numpy as np
from stitcher.warp import canvas_bounds, canvas_bounds_n


def test_identity_canvas_is_image_size():
    xmin, ymin, W, H = canvas_bounds((100, 200, 3), (100, 200, 3), np.eye(3))
    assert (xmin, ymin, W, H) == (0, 0, 200, 100)


def test_shifted_image_extends_canvas():
    # a point at x in the reference frame is at x - 50 in the other image,
    # so the other image sits 50 pixels to the right of the reference
    Hs = np.array([[1, 0, -50], [0, 1, 0], [0, 0, 1]], float)
    xmin, ymin, W, H = canvas_bounds((100, 200, 3), (100, 200, 3), Hs)
    assert (xmin, ymin, W, H) == (0, 0, 250, 100)

def test_canvas_bounds_n_three_shifted_images():
    def shift(dx):
        return np.array([[1, 0, dx], [0, 1, 0], [0, 0, 1]], float)
    # image 0 is 50 px left of the reference, image 2 is 50 px right
    ref_to = [shift(50), np.eye(3), shift(-50)]
    assert canvas_bounds_n([(100, 200, 3)] * 3, ref_to) == (-50, 0, 300, 100)