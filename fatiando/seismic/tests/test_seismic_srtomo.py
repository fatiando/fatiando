from __future__ import division, print_function, absolute_import
from future.builtins import range
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
from pytest import raises

from fatiando.seismic import srtomo
from fatiando.mesher import Square, SquareMesh
from fatiando.seismic import ttime2d


def test_general():
    """
    General test of the class SRTomo, as it is in the docs of this class.
    """
    model = SquareMesh((0, 10, 0, 10), shape=(2, 1), props={'vp': [2., 5.]})
    src = (5, 0)
    srcs = [src, src]
    recs = [(0, 0), (5, 10)]
    ttimes = ttime2d.straight(model, 'vp', srcs, recs)
    mesh = SquareMesh((0, 10, 0, 10), shape=(2, 1))
    tomo = srtomo.SRTomo(ttimes, srcs, recs, mesh)
    assert_array_almost_equal(tomo.fit().estimate_, np.array([2., 5.]), 9)


def test_jacobian():
    """
    srtomo.SRTomo.jacobian return the jacobian of the model provided. In this
    simple model, the jacobian can be easily calculated.
    """
    model = SquareMesh((0, 10, 0, 10), shape=(2, 1), props={'vp': [2., 5.]})
    src = (5, 0)
    srcs = [src, src]
    recs = [(0, 0), (5, 10)]
    ttimes = ttime2d.straight(model, 'vp', srcs, recs)
    mesh = SquareMesh((0, 10, 0, 10), shape=(2, 1))
    tomo = srtomo.SRTomo(ttimes, srcs, recs, mesh)
    assert_array_almost_equal(tomo.jacobian().todense(),
                              np.array([[5., 0.], [5., 5.]]), 9)


def test_predicted():
    """
    Test to verify srtomo.SRTomo.predicted function. Given the correct
    parameters, this function must return the result of the forward data.
    """
    model = SquareMesh((0, 10, 0, 10), shape=(2, 1), props={'vp': [2., 5.]})
    src = (5, 0)
    srcs = [src, src]
    recs = [(0, 0), (5, 10)]
    ttimes = ttime2d.straight(model, 'vp', srcs, recs)
    mesh = SquareMesh((0, 10, 0, 10), shape=(2, 1))
    tomo = srtomo.SRTomo(ttimes, srcs, recs, mesh)
    # The parameter used inside the class is slowness, so 1/vp.
    tomo.p_ = np.array([1./2., 1./5.])
    assert_array_almost_equal(tomo.predicted(), ttimes, 9)
