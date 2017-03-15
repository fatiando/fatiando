from __future__ import absolute_import
from ...mesher import Square


def test_square_copy():
    orig = Square([0, 1, 2, 4], {'density': 750})
    cp = orig.copy()
    assert isinstance(orig, Square)
    assert isinstance(cp, Square)
    assert orig.x1 == cp.x1
    assert orig.x2 == cp.x2
    assert orig.y1 == cp.y1
    assert orig.y2 == cp.y2
    assert orig.props == cp.props
    cp.addprop('magnetization', 10)
    assert orig.props != cp.props
    assert orig.props['density'] == cp.props['density']
    cp.props['density'] = 850
    assert orig.props['density'] != cp.props['density']
    assert orig.nverts == cp.nverts
    assert cp is not orig
