from __future__ import absolute_import
from ...mesher import Sphere
import numpy as np


def test_sphere_copy():
    orig = Sphere(1, 2, 3, 10, {'density': 3000})
    cp = orig.copy()
    assert orig is not cp
    assert orig.x == cp.x
    assert orig.y == cp.y
    assert orig.z == cp.z
    assert orig.props == cp.props
    assert np.array_equal(orig.center, cp.center)

    cp.x = 4
    cp.y = 6
    cp.z = 7
    assert orig.x != cp.x
    assert orig.y != cp.y
    assert orig.z != cp.z
    cp.props['density'] = 2000
    assert orig.props['density'] != cp.props['density']
