from ...mesher import Tesseroid


def test_tesseroid_copy():
    t = Tesseroid(1, 2, 3, 4, 6, 5, {'density': 200})
    cp = t.copy()
    assert t.get_bounds() == cp.get_bounds()
    assert t.props == cp.props
    t.props['density'] = 599
    assert t.props['density'] != cp.props['density']
    assert t is not cp
