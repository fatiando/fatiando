from __future__ import absolute_import
from ...mesher import PolygonalPrism

import numpy as np


def test_polygonal_prism_copy():
    verts = [[1, 1], [1, 2], [2, 2], [2, 1]]
    orig = PolygonalPrism(verts, 0, 100)
    orig.addprop('density', 2670)
    copy = orig.copy()
    assert orig.nverts == copy.nverts
    assert orig.props == copy.props
    assert np.array_equal(orig.x, copy.x)
    assert np.array_equal(orig.y, copy.y)
    assert orig.z1 == copy.z1
    assert orig.z2 == copy.z2
    assert orig is not copy
