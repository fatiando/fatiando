from __future__ import division
import numpy as np

from fatiando.constants import MEAN_EARTH_RADIUS
from fatiando.gravmag import spherical_shell


def test_shell_against_published():
    "gravmag.spherical_shell is compatible with published results"
    # Compare the computed effects of the spherical shell with the ones
    # published in:
    # Grombein, T., K. Seitz, and B. Heck (2013), Optimized formulas for the
    # gravitational field of a tesseroid, J Geod, 87(7), 645–660,
    # doi:10.1007/s00190-013-0636-1.
    pass
