"""
Load data/models from images.
"""
from __future__ import absolute_import
import os
from PIL import Image
import scipy.misc
import numpy as np

from . import check_hash


# Get the file path of some sample images to use in tests and in the gallery.
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

SAMPLE_IMAGE_SHA256 = \
    'fc4e0dbfa5cf55995de28a836d0ee52021e7726cdf306fb6c3bb1df874ddb785'
SAMPLE_IMAGE = os.path.join(DATA_DIR, 'sample-image.png')
check_hash(SAMPLE_IMAGE, known_hash=SAMPLE_IMAGE_SHA256, hash_type='sha256')

SAMPLE_IMAGE_SMALL_SHA256 = \
    '65234acf3ebd3e12fa7b3567f1b2ebdb18421fd54b4300de328ddbf2a8142325'
SAMPLE_IMAGE_SMALL = os.path.join(DATA_DIR, 'sample-image-small.png')
check_hash(SAMPLE_IMAGE_SMALL, known_hash=SAMPLE_IMAGE_SMALL_SHA256,
           hash_type='sha256')


def from_image(fname, pixel_thresh=10, return_colors=False):
    """
    Create a model template from an image file.

    The template is a 2D array with integers encoding each different color of
    an image. The numbers are ordered by the number of pixels each color has.
    So the most abundant color will be assigned the value 0.

    The template can be used to create a model by replacing the each integer
    with the desired physical property value, for example
    ``model = template.copy(); model[template == 0] = 2670``.

    A few tips for better results:

    * Use images with a few sharp colors.
    * Avoid smoothing effects because they create a gradient of colors, so each
      color in the gradient will be flagged as a unique value in the template.
    * When scaling an image, use nearest neighbors interpolation.
    * Use the "pencil" tool to draw the image rather than the "brush" tool to
      avoid color gradients.

    Parameters:

    * fname : string
        The name of the image file. Can be any format supported by the Python
        imaging library, though PNG is recommended.
    * pixel_thresh : int
        Colors with less pixels than *pixel_thresh* will be ignored in the
        template. They will be assigned the value 0.
    *  return_colors : True or False
        If True, will return a list of the colors corresponding to each value
        of the template.

    Returns:

    * template : 2D array
        The image but replacing the colors with integers starting from 0.
    * colors : list
        List of the colors corresponding to each value of the template. Colors
        are tuples of RGBA values.

    Examples:

    >>> import fatiando.datasets as ds
    >>> template = ds.from_image(ds.SAMPLE_IMAGE_SMALL)
    >>> print(template)
    [[2 2 2 2 2 2 2 0 0 3 0 0 0 2 2 2 2 2 2 2]
     [2 2 2 2 2 1 1 0 3 3 0 0 0 1 1 2 2 2 2 2]
     [2 2 2 2 1 0 0 0 0 0 0 0 0 0 0 0 1 2 2 2]
     [2 2 1 1 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2]
     [2 2 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2]
     [2 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 2]
     [2 0 1 0 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 2]
     [0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1]
     [0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1]
     [0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1]
     [0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1]
     [0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1]
     [0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1]
     [2 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 2]
     [2 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 2]
     [2 2 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 2 2]
     [2 2 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 2 2]
     [2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2]
     [2 2 2 2 2 0 0 0 3 0 0 0 0 0 0 2 2 2 2 2]
     [2 2 2 2 2 2 2 3 3 3 3 3 3 2 2 2 2 2 2 2]]
    >>> # If you squint you can see the Earth in there. (Sure you can, right?)
    >>> # Assign physical properties to each value in the template, use numpy
    >>> # fancy/boolean indexing.
    >>> model = template.copy()
    >>> model[template==1] = 0
    >>> model[template==0] = 1
    >>> model[template==2] = 8
    >>> print(model)
    [[8 8 8 8 8 8 8 1 1 3 1 1 1 8 8 8 8 8 8 8]
     [8 8 8 8 8 0 0 1 3 3 1 1 1 0 0 8 8 8 8 8]
     [8 8 8 8 0 1 1 1 1 1 1 1 1 1 1 1 0 8 8 8]
     [8 8 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 8 8]
     [8 8 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 8 8]
     [8 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 8]
     [8 1 0 1 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 8]
     [1 1 1 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0]
     [1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0]
     [1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0]
     [1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0]
     [1 1 1 1 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0]
     [1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0]
     [8 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 0 1 8]
     [8 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 8]
     [8 8 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 8 8]
     [8 8 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 8 8]
     [8 8 8 1 1 1 1 1 1 1 1 1 1 1 1 1 1 8 8 8]
     [8 8 8 8 8 1 1 1 3 1 1 1 1 1 1 8 8 8 8 8]
     [8 8 8 8 8 8 8 3 3 3 3 3 3 8 8 8 8 8 8 8]]
    >>> # You can see the colors that make up the image as well
    >>> template, colors = ds.from_image(ds.SAMPLE_IMAGE_SMALL,
    ...                                  return_colors=True)
    >>> print(colors)
    [[  0   0 255 255]
     [  0 255   0 255]
     [123 123 123 255]
     [255 255 255 255]]
    >>> # The colors are RGBA values from 0-255.
    >>> # The most abundant color is a blue (255 in the third column),
    >>> # followed by a green, a grey, and finally a white.

    """
    assert pixel_thresh >= 0, "pixel_thresh must be > 0"
    img = Image.open(fname)
    data = scipy.misc.fromimage(img)
    count, colors = zip(*[[n, c] for n, c in img.getcolors()
                          if n >= pixel_thresh])
    sort = np.argsort(count)[::-1]
    colors = np.array(colors)[sort]
    template = np.zeros(data.shape[:2], dtype=np.int)
    for i, c in enumerate(colors):
        template[np.all(data == c, axis=2)] = i
    # For now, any pixel not of the top colors (eliminated by pixel_thresh) is
    # assigned index 0. A better way would be interpolate using nearest
    # neighbors
    if return_colors:
        return template, colors
    else:
        return template
