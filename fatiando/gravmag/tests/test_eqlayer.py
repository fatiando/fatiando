from __future__ import division, absolute_import
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
from ..eqlayer import EQLGravity, EQLTotalField, PELGravity, PELTotalField, \
    PELSmoothness
from ...inversion import Damping
from .. import sphere, prism
from ...mesher import PointGrid, Prism
from ... import utils, gridder


@pytest.mark.xfail
def test_pel_polereduce():
    "PELTotalField can reduce data to the pole"
    # Use remanent magnetization
    sinc, sdec = -80, 5
    model = [Prism(-100, 100, -500, 500, 0, 100,
                   {'magnetization': utils.ang2vec(5, sinc, sdec)})]
    inc, dec = -83, 5
    shape = (40, 40)
    area = [-2000, 2000, -2000, 2000]
    x, y, z = gridder.regular(area, shape, z=-100)
    data = prism.tf(x, y, z, model, inc, dec)
    true = prism.tf(x, y, z, model, -90, 0, pmag=utils.ang2vec(5, -90, 0))

    layer = PointGrid(area, 100, shape)
    windows = (20, 20)
    degree = 3
    pel = PELTotalField(x, y, z, data, inc, dec, layer, windows, degree,
                        sinc, sdec)
    eql = pel + 1e-22*PELSmoothness(layer, windows, degree)
    eql.fit()

    assert_allclose(eql[0].predicted(), data, rtol=0.06)

    layer.addprop('magnetization',
                  utils.ang2vec(eql.estimate_, inc=-90, dec=0))
    calc = sphere.tf(x, y, z, layer, inc=-90, dec=0)

    assert_allclose(calc, true, atol=10, rtol=0.05)


def test_pelgrav_prism_interp():
    "PELGravity can interpolate data from a prism"
    model = [Prism(-300, 300, -500, 500, 100, 600, {'density': 400})]
    shape = (40, 40)
    n = shape[0]*shape[1]
    area = [-2000, 2000, -2000, 2000]
    x, y, z = gridder.scatter(area, n, z=-100, seed=42)
    data = prism.gz(x, y, z, model)

    layer = PointGrid(area, 100, shape)
    windows = (20, 20)
    degree = 1
    eql = (PELGravity(x, y, z, data, layer, windows, degree) +
           5e-22*PELSmoothness(layer, windows, degree))
    eql.fit()
    layer.addprop('density', eql.estimate_)

    assert_allclose(eql[0].predicted(), data, rtol=0.01)

    xp, yp, zp = gridder.regular(area, shape, z=-100)
    true = prism.gz(xp, yp, zp, model)
    calc = sphere.gz(xp, yp, zp, layer)

    assert_allclose(calc, true, atol=0.001, rtol=0.05)


def test_eql_grav_jacobian():
    "EQLGravity produces the right Jacobian matrix for single source"
    model = PointGrid([-10, 10, -10, 10], 500, (2, 2))[0]
    model.addprop('density', 1)
    n = 1000
    x, y, z = gridder.scatter([-10, 10, -10, 10], n, z=-100, seed=42)
    data = sphere.gz(x, y, z, [model])

    eql = EQLGravity(x, y, z, data, [model])
    A = eql.jacobian(None)

    assert A.shape == (n, 1)
    assert_allclose(A[:, 0], data, rtol=0.01)


def test_eql_mag_jacobian():
    "EQLTotalField produces the right Jacobian matrix for single source"
    inc, dec = -30, 20
    model = PointGrid([-10, 10, -10, 10], 500, (2, 2))[0]
    model.addprop('magnetization', utils.ang2vec(1, inc, dec))
    n = 1000
    x, y, z = gridder.scatter([-10, 10, -10, 10], n, z=-100, seed=42)
    data = sphere.tf(x, y, z, [model], inc, dec)

    eql = EQLTotalField(x, y, z, data, inc, dec, [model])
    A = eql.jacobian(None)

    assert A.shape == (n, 1)
    assert_allclose(A[:, 0], data, rtol=0.01)


def test_eqlgrav_prism_interp():
    "EQLGravity can interpolate data from a prism"
    model = [Prism(-300, 300, -500, 500, 100, 600, {'density': 400})]
    shape = (30, 30)
    n = shape[0]*shape[1]
    area = [-2000, 2000, -2000, 2000]
    x, y, z = gridder.scatter(area, n, z=-100, seed=42)
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
    eql = (EQLTotalField(x, y, z, data, inc, dec, layer, sinc, sdec) +
           1e-24*Damping(layer.size))
    eql.fit()

    assert_allclose(eql[0].predicted(), data, rtol=0.01)

    layer.addprop('magnetization',
                  utils.ang2vec(eql.estimate_, inc=-90, dec=0))
    calc = sphere.tf(x, y, z, layer, inc=-90, dec=0)

    assert_allclose(calc, true, atol=10, rtol=0.05)
