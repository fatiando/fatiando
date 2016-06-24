from __future__ import division
import numpy as np
from numpy.testing import assert_allclose
from fatiando.gravmag.eqlayer import EQLGravity, EQLTotalField, \
                                     PELGravity, PELTotalField
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
    max_diff = np.abs(calc - true).max()
    assert_allclose(calc, true, rtol=0.05,
                    err_msg='max diff: {}'.format(max_diff))



