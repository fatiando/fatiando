from __future__ import division
import numpy as np
from numpy.testing import assert_allclose
from fatiando.gravmag import transform, prism
from fatiando import gridder, utils
from fatiando.mesher import Prism

def test_potential_field_tilt():
    "gravmag.transform tilt returns sane values"
    model = [Prism(-1000, 1000, -500, 500, 0, 2000, {'density': 200})]
    shape = (300, 300)
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], shape, z=-100)
    inc, dec = -30, 0
    potential = prism.tf(x, y, z, model, inc, dec)
    tilt_angle = tilt(x, y, potential, shape)
    assert np.degree(tilt)
