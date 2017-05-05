"""
Test the polygonal prism forward modeling against the prism forward modeling.
We can make a polygonal prism with 4 vertices that is exactly the same as a
rectangular prism. The potential fields should be the same in both cases.
"""
from __future__ import absolute_import, division
import numpy as np
import numpy.testing as npt
import pytest

from ...mesher import PolygonalPrism, Prism
from .. import polyprism, prism
from ... import utils, gridder


FIELDS = 'gz gxx gxy gxz gyy gyz gzz tf bx by bz'.split()
KERNELS = ['kernel' + k for k in 'xx xy xz yy yz zz'.split()]


@pytest.fixture(scope="module")
def models():
    """
    Equivalent prism and polygonal prism models for testing.
    """
    props1 = {'density': 2., 'magnetization': utils.ang2vec(-20, 25, -10)}
    props2 = {'density': -3., 'magnetization': utils.ang2vec(10, -30, 50)}
    polymodel = [
        PolygonalPrism([[100, -100], [100, 100], [-100, 100], [-100, -100]],
                       100, 300, props1),
        PolygonalPrism([[400, -100], [600, -100], [600, 100], [400, 100]], 100,
                       300, props2)
    ]
    prismmodel = [Prism(-100, 100, -100, 100, 100, 300, props1),
                  Prism(400, 600, -100, 100, 100, 300, props2)]
    return polymodel, prismmodel


@pytest.fixture(scope="module")
def grid():
    "The computation grid"
    return gridder.regular([-500, 1000, -500, 1000], (71, 81), z=-1)


def test_polyprism_vs_prism(grid, models):
    "gravmag polyprism and prism forward modeling have compatible results"
    inc, dec = -30, 50
    x, y, z = grid
    polymodel, prismmodel = models
    for field in FIELDS:
        if field == 'tf':
            resprism = getattr(prism, field)(x, y, z, prismmodel, inc, dec)
            respoly = getattr(polyprism, field)(x, y, z, polymodel, inc, dec)
        else:
            resprism = getattr(prism, field)(x, y, z, prismmodel)
            respoly = getattr(polyprism, field)(x, y, z, polymodel)
        if field in 'bx by bz'.split():
            tolerance = 1e-4
        elif field == 'tf':
            tolerance = 1e-5
        else:
            tolerance = 1e-8
        npt.assert_allclose(respoly, resprism, atol=tolerance, rtol=0,
                            err_msg='field: {}'.format(field))
    for kernel in KERNELS:
        for i in range(len(polymodel)):
            resprism = getattr(prism, kernel)(x, y, z, prismmodel[i])
            respoly = getattr(polyprism, kernel)(x, y, z, polymodel[i])
            tolerance = 1e-8
            max_diff = np.abs(respoly - resprism).max()
            max_val = np.abs(respoly).max()
            msg = 'kernel: {}  max diff: {}  max val: {}'.format(
                kernel, max_diff, max_val)
            npt.assert_allclose(respoly, resprism, atol=tolerance, rtol=0,
                                err_msg=msg)
