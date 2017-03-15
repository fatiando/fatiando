from __future__ import absolute_import
import os
import numpy as np
import numpy.testing as npt

from .. import from_image


TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def test_from_image():
    "Test if colors are in the correct order and places"
    test_img = os.path.join(TEST_DATA_DIR, 'test-image.png')
    # Use a pixel_thresh of 5 to ignore the green points
    template, colors = from_image(test_img, return_colors=True, pixel_thresh=5)
    true_colors = np.array([[0, 0, 0, 255],     # black
                            [0, 0, 255, 255],   # blue
                            [255, 0, 0, 255]])  # red
    npt.assert_equal(colors, true_colors)
    npt.assert_equal(template[:28, :], 0)
    npt.assert_equal(template[28:, 38:], 1)
    npt.assert_equal(template[28:, :38], 2)
    # Read the green pixels as well
    template, colors = from_image(test_img, return_colors=True, pixel_thresh=1)
    true_colors = np.array([[0, 0, 0, 255],     # black
                            [0, 0, 255, 255],   # blue
                            [255, 0, 0, 255],   # red
                            [0, 255, 0, 255]])  # green
    npt.assert_equal(colors, true_colors)
    assert template[template == 3].size == 4, 'Number of green pixels not 4'
    # Check if the green pixels are in the right place
    true_coords = np.array([[0, 99], [11, 37], [27, 0], [27, 99]])
    green_coords = np.transpose(np.where(template == 3))
    npt.assert_equal(green_coords, true_coords)
