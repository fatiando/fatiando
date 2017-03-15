from __future__ import absolute_import
from ...mesher import Polygon


def test_polygon_copy():
    pol = Polygon([[0, 0], [1, 4], [2, 5]], {'density': 500})
    cp = pol.copy()
    assert pol.nverts == cp.nverts
    assert pol.vertices.all() == cp.vertices.all()
    assert pol is not cp
