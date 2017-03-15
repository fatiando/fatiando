from __future__ import absolute_import
from ...mesher import GeometricElement


def test_geometric_element_copy():
    orig = GeometricElement({'density': 5000})
    cp = orig.copy()
    assert orig.props == cp.props
    assert orig is not cp
    orig.props['density'] = 3000
    assert orig.props != cp.props
