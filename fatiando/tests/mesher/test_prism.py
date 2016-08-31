from ...mesher import Prism


def test_prism_copy():
    p = Prism(1, 2, 3, 4, 5, 6, {'density': 200})
    cp = p.copy()
    assert p.props == cp.props
    assert p is not cp
    assert p.x1 == cp.x1
    assert p.x2 == cp.x2
    assert p.y1 == cp.y1
    assert p.y2 == cp.y2
    assert p.z1 == cp.z1
    assert p.z2 == cp.z2
    p.x1 = 22
    assert p.x1 != cp.x1
    p.props['density'] = 399
    assert p.props['density'] != cp.props['density']
