from __future__ import division
import numpy as np
from numpy.testing import assert_allclose
from fatiando.gravmag.eqlayer import (EQLGravity, EQLTotalField,
                                      PELGravity, PELTotalField)
from fatiando.inversion import Damping
from fatiando.gravmag import sphere, prism
from fatiando.mesher import PointGrid, Prism
from fatiando import utils, gridder


def test_eqlgrav_prism_interp():
    "EQLGravity can interpolate data from a prism"
    model = [Prism(-300, 300, -500, 500, 100, 600, {'density': 400})]
    shape = (30, 30)
    n = shape[0]*shape[1]
    area = [-2000, 2000, -2000, 2000]
    x, y, z = gridder.scatter(area,  n, z=-100, seed=42)
    data = prism.gz(x, y, z, model)
    layer = PointGrid(area, 200, shape)
    eql = EQLGravity(x, y, z, data, layer) + 1e-23*Damping(layer.size)
    eql.fit()
    layer.addprop('density', eql.estimate_)

    assert_allclose(eql[0].predicted(), data, rtol=0.01)

    xp, yp, zp = gridder.regular(area, shape, z=-100)
    true = prism.gz(xp, yp, zp, model)
    calc = sphere.gz(xp, yp, zp, layer)

    assert_allclose(calc, true, rtol=0.05)


def test_eqlayer_polereduce():
    "EQLTotalField can reduce data to the pole"
    # Use remanent magnetization
    sinc, sdec = -70, 30
    model = [Prism(-100, 100, -500, 500, 0, 100,
                   {'magnetization': utils.ang2vec(5, sinc, sdec)})]
    inc, dec = -60, -15
    shape = (50, 50)
    area = [-2000, 2000, -2000, 2000]
    x, y, z = gridder.regular(area, shape, z=-100)
    data = prism.tf(x, y, z, model, inc, dec)
    true = prism.tf(x, y, z, model, -90, 0, pmag=utils.ang2vec(5, -90, 0))

    layer = PointGrid(area, 200, shape)
    eql = (EQLTotalField(x, y, z, data, inc, dec, layer, sinc, sdec)
           + 1e-24*Damping(layer.size))
    eql.fit()

    assert_allclose(eql[0].predicted(), data, rtol=0.01)

    layer.addprop('magnetization',
                  utils.ang2vec(eql.estimate_, inc=-90, dec=0))
    calc = sphere.tf(x, y, z, layer, inc=-90, dec=0)

    assert_allclose(calc, true, atol=10, rtol=0.05)

