"""
Test the polygonal prism forward modeling against the prism forward modeling.
We can make a polygonal prism with 4 vertices that is exactly the same as a
rectangular prism. The potential fields should be the same in both cases.
"""
from __future__ import absolute_import, division
import numpy as np
import numpy.testing as npt

from ...mesher import PolygonalPrism, Prism
from .. import polyprism, prism, _polyprism_numpy
from ... import utils, gridder

inc, dec = -30, 50
props1 = {'density': 2., 'magnetization': utils.ang2vec(-20, 25, -10)}
props2 = {'density': -3., 'magnetization': utils.ang2vec(10, inc, dec)}
model = [
    PolygonalPrism([[100, -100], [100, 100], [-100, 100], [-100, -100]], 100,
                   300, props1),
    PolygonalPrism([[400, -100], [600, -100], [600, 100], [400, 100]], 100,
                   300, props2)
]
prismmodel = [Prism(-100, 100, -100, 100, 100, 300, props1),
              Prism(400, 600, -100, 100, 100, 300, props2)]
xp, yp, zp = gridder.regular([-500, 1000, -500, 1000], (71, 71), z=-1)


def test_polyprism_vs_prism():
    "gravmag polyprism and prism forward modeling have compatible results"
    # Polyprism doesn't support all fields that prism supports
    fields = 'gz gxx gxy gxz gyy gyz gzz tf bx by bz'.split()
    for field in fields:
        if field == 'tf':
            resprism = getattr(prism, field)(xp, yp, zp, prismmodel, inc, dec)
            respoly = getattr(polyprism, field)(xp, yp, zp, model, inc, dec)
        else:
            resprism = getattr(prism, field)(xp, yp, zp, prismmodel)
            respoly = getattr(polyprism, field)(xp, yp, zp, model)
        if field in 'bx by bz'.split():
            tolerance = 1e-4
        elif field == 'tf':
            tolerance = 1e-5
        else:
            tolerance = 1e-8
        npt.assert_allclose(respoly, resprism, atol=tolerance, rtol=0,
                            err_msg='field: {}'.format(field))
