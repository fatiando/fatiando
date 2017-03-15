"""
Test the geometry primitive classes.
"""
from __future__ import division, absolute_import
from .. import Polygon, Square, Prism, PolygonalPrism, Tesseroid, Sphere
import numpy as np
import numpy.testing as npt


def test_polygon_copy():
    "Make sure copy method works, even for props dictionary"
    pol = Polygon([[0, 0], [1, 4], [2, 5]], {'density': 500})
    cp = pol.copy()
    assert pol is not cp
    assert pol.nverts == cp.nverts
    npt.assert_equal(pol.vertices, cp.vertices)


def test_polygonal_prism_copy():
    "Make sure copy method works, even for props dictionary"
    verts = [[1, 1], [1, 2], [2, 2], [2, 1]]
    orig = PolygonalPrism(verts, 0, 100)
    orig.addprop('density', 2670)
    copy = orig.copy()
    assert orig.nverts == copy.nverts
    assert orig.props == copy.props
    npt.assert_equal(orig.x, copy.x)
    npt.assert_equal(orig.y, copy.y)
    assert orig.z1 == copy.z1
    assert orig.z2 == copy.z2
    assert orig is not copy


def test_prism_copy():
    "Make sure copy method works, even for props dictionary"
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
    npt.assert_equal(p.bounds, cp.bounds)
    npt.assert_equal(p.center, cp.center)
    p.x1 = 22
    assert p.x1 != cp.x1
    p.props['density'] = 399
    assert p.props['density'] != cp.props['density']


def test_sphere_copy():
    "Make sure copy method works, even for props dictionary"
    orig = Sphere(1, 2, 3, 10, {'density': 3000})
    cp = orig.copy()
    assert orig is not cp
    assert orig.x == cp.x
    assert orig.y == cp.y
    assert orig.z == cp.z
    assert orig.props == cp.props
    npt.assert_equal(orig.center, cp.center)
    cp.x = 4
    cp.y = 6
    cp.z = 7
    assert orig.x != cp.x
    assert orig.y != cp.y
    assert orig.z != cp.z
    cp.props['density'] = 2000
    assert orig.props['density'] != cp.props['density']


def test_square_copy():
    "Make sure copy method works, even for props dictionary"
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
    npt.assert_equal(orig.x, cp.x)
    npt.assert_equal(orig.y, cp.y)
    npt.assert_equal(orig.vertices, cp.vertices)


def test_tesseroid_copy():
    "Make sure copy method works, even for props dictionary"
    t = Tesseroid(1, 2, 3, 4, 6, 5, {'density': 200})
    cp = t.copy()
    npt.assert_equal(t.bounds, cp.bounds)
    npt.assert_equal(t.center, cp.center)
    assert t.props == cp.props
    t.props['density'] = 599
    assert t.props['density'] != cp.props['density']
    assert t is not cp
