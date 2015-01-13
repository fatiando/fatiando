from __future__ import division
from fatiando.seismic.wavefd import (Ricker,
                                     Gauss,
                                     ElasticSH)

import numpy as np
from numpy.testing import assert_almost_equal
import os

def test_Sources():
    'testing simple source parameters'
    w = Ricker(amp=10, cf=20)
    assert (w(0) == 10.0)  # maximum
    assert_almost_equal(w(-2), w(2), decimal=5, err_msg="Ricker has no symmetry")
    w = Gauss(amp=10, cf=20)
    assert (w(0) == 0.0)  # maximum
    assert_almost_equal(w(-2), w(2), decimal=5, err_msg="Gauss has no symmetry")


def test_wavefd_elasticsh_run():
    'make a simple run of elastic sh'
    shape = (50, 50)
    velocity = 1500*np.ones(shape)
    density = 2200*np.ones(shape)
    sim = ElasticSH(velocity, density, (5, 5))
    sim.add_point_source((0, shape[1]//2), Ricker(5, 7., 1./7.))
    sim.run(100)