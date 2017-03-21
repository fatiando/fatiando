"""
Test the geometric element classes.
"""
from __future__ import absolute_import, division
import numpy as np
from numpy.testing import assert_almost_equal
from pytest import raises

from ..geometry import Square
from ..geometry import GeometricElement
from ..geometry import Polygon
from ..geometry import PolygonalPrism
from ..geometry import Prism
from ..geometry import Sphere
from ..geometry import Tesseroid
from ..geometry import TriaxialEllipsoid
from ..geometry import ProlateEllipsoid
from ..geometry import OblateEllipsoid
from ..geometry import _coord_transf_matrix_triaxial
from ..geometry import _coord_transf_matrix_oblate
from ..geometry import _auxiliary_angles


def test_triaxial_ellipsoid_copy():
    'Check the elements of a duplicated ellipsoid'
    orig = TriaxialEllipsoid(12, 42, 53, 61, 35, 14, 10, 20, 30, {
                             'remanent magnetization': [10, 25, 40],
                             'principal susceptibilities': [0.562, 0.485,
                                                            0.25],
                             'susceptibility angles': [90, 0, 0]})
    cp = orig.copy()
    assert orig is not cp
    assert orig.x == cp.x
    assert orig.y == cp.y
    assert orig.z == cp.z
    assert orig.large_axis == cp.large_axis
    assert orig.intermediate_axis == cp.intermediate_axis
    assert orig.small_axis == cp.small_axis
    assert orig.props == cp.props

    cp.x = 4
    cp.y = 6
    cp.z = 7
    cp.props['principal susceptibilities'] = [0.7, 0.9, 1]
    assert orig.x != cp.x
    assert orig.y != cp.y
    assert orig.z != cp.z
    assert orig.props['principal susceptibilities'] != \
        cp.props['principal susceptibilities']


def test_triaxial_ellipsoid_axes():
    'axes must be given in descending order'
    raises(AssertionError, TriaxialEllipsoid, x=1, y=2, z=3,
           large_axis=6, intermediate_axis=5, small_axis=14,
           strike=10, dip=20, rake=30)


def test_triaxial_ellipsoid_principal_susceptibilities_fmt():
    'principal susceptibilities must be a list containing 3 elements'
    e = TriaxialEllipsoid(x=1, y=2, z=3, large_axis=6,
                          intermediate_axis=5, small_axis=4,
                          strike=10, dip=20, rake=30,
                          props={'remanent magnetization': [10, 25, 40],
                                 'principal susceptibilities': [0.562, 0.485],
                                 'susceptibility angles': [90, 0, 0]})

    with raises(AssertionError):
        e.susceptibility_tensor


def test_triaxial_ellipsoid_susceptibility_angles_fmt():
    'susceptibility angles must be a list containing 3 elements'
    e = TriaxialEllipsoid(x=1, y=2, z=3, large_axis=6,
                          intermediate_axis=5, small_axis=4,
                          strike=10, dip=20, rake=30,
                          props={'remanent magnetization': [10, 25, 40],
                                 'principal susceptibilities': [0.562, 0.485,
                                                                0.2],
                                 'susceptibility angles': [90, 0]})

    with raises(AssertionError):
        e.susceptibility_tensor


def test_triaxial_ellipsoid_susceptibility_tensor_symm():
    'susceptibility tensor must be symmetric'
    e = TriaxialEllipsoid(x=1, y=2, z=3, large_axis=6,
                          intermediate_axis=5, small_axis=4,
                          strike=10, dip=20, rake=30,
                          props={'remanent magnetization': [10, 25, 40],
                                 'principal susceptibilities': [0.562, 0.485,
                                                                0.25],
                                 'susceptibility angles': [90, 0, 0]})
    assert_almost_equal(e.susceptibility_tensor, e.susceptibility_tensor.T,
                        decimal=15)


def test_triaxial_ellipsoid_principal_susceptibilities_order():
    'principal susceptibilities must be given in descending order'
    e = TriaxialEllipsoid(x=1, y=2, z=3, large_axis=6,
                          intermediate_axis=5, small_axis=4,
                          strike=10, dip=20, rake=30,
                          props={'remanent magnetization': [10, 25, 40],
                                 'principal susceptibilities': [0.562, 0.185,
                                                                0.25],
                                 'susceptibility angles': [90, 0, 0]})
    with raises(AssertionError):
        e.susceptibility_tensor


def test_triaxial_ellipsoid_principal_susceptibilities_signal():
    'principal susceptibilities must be all positive'
    e = TriaxialEllipsoid(x=1, y=2, z=3, large_axis=6,
                          intermediate_axis=5, small_axis=4,
                          strike=10, dip=20, rake=30,
                          props={'remanent magnetization': [10, 25, 40],
                                 'principal susceptibilities': [0.562, 0.485,
                                                                -0.25],
                                 'susceptibility angles': [90, 0, 0]})
    with raises(AssertionError):
        e.susceptibility_tensor


def test_coord_transf_matrix_triaxial_known():
    'Coordinate transformation matrix built with known orientation angles'
    alpha = -np.pi
    gamma = np.pi/2
    delta = 0
    transf_matrix = _coord_transf_matrix_triaxial(alpha, gamma, delta)
    assert_almost_equal(transf_matrix, np.identity(3), decimal=15)


def test_coord_transf_matrix_triaxial_orthogonal():
    'Coordinate transformation matrix must be orthogonal'
    alpha = 38.9
    gamma = -0.2
    delta = 174
    transf_matrix = _coord_transf_matrix_triaxial(alpha, gamma, delta)
    dot1 = np.dot(transf_matrix, transf_matrix.T)
    dot2 = np.dot(transf_matrix.T, transf_matrix)
    assert_almost_equal(dot1, dot2, decimal=15)
    assert_almost_equal(dot1, np.identity(3), decimal=15)
    assert_almost_equal(dot2, np.identity(3), decimal=15)


def test_prolate_ellipsoid_copy():
    'Check the elements of a duplicated ellipsoid'
    orig = ProlateEllipsoid(31, 2, 83, 56, 54, 1, 29, 70,
                            props={'remanent magnetization': [10, 25, 40],
                                   'principal susceptibilities': [0.562, 0.485,
                                                                  -0.25],
                                   'susceptibility angles': [90, 0, 0]})
    cp = orig.copy()
    assert orig is not cp
    assert orig.x == cp.x
    assert orig.y == cp.y
    assert orig.z == cp.z
    assert orig.large_axis == cp.large_axis
    assert orig.small_axis == cp.small_axis
    assert orig.props == cp.props

    cp.x = 4
    cp.y = 6
    cp.z = 7
    cp.props['principal susceptibilities'] = [0.7, 0.9, 1]
    assert orig.x != cp.x
    assert orig.y != cp.y
    assert orig.z != cp.z
    assert orig.props['principal susceptibilities'] != \
        cp.props['principal susceptibilities']


def test_prolate_ellipsoid_axes():
    'axes must be given in descending order'
    raises(AssertionError, ProlateEllipsoid, x=1, y=2, z=3,
           large_axis=2, small_axis=4, strike=10, dip=20, rake=30)


def test_prolate_ellipsoid_principal_susceptibilities_fmt():
    'principal susceptibilities must be a list containing 3 elements'
    e = ProlateEllipsoid(x=1, y=2, z=3,
                         large_axis=6, small_axis=4,
                         strike=10, dip=20, rake=30,
                         props={'remanent magnetization': [10, 25, 40],
                                'principal susceptibilities': [0.562, 0.485],
                                'susceptibility angles': [90, 0, 0]})

    with raises(AssertionError):
        e.susceptibility_tensor


def test_prolate_ellipsoid_susceptibility_angles_fmt():
    'susceptibility angles must be a list containing 3 elements'
    e = ProlateEllipsoid(x=1, y=2, z=3,
                         large_axis=6, small_axis=4,
                         strike=10, dip=20, rake=30,
                         props={'remanent magnetization': [10, 25, 40],
                                'principal susceptibilities': [0.562, 0.485,
                                                               0.2],
                                'susceptibility angles': [90, 0]})

    with raises(AssertionError):
        e.susceptibility_tensor


def test_prolate_ellipsoid_principal_susceptibilities_order():
    'principal susceptibilities must be given in descending order'
    e = ProlateEllipsoid(x=1, y=2, z=3, large_axis=6, small_axis=4,
                         strike=10, dip=20, rake=30,
                         props={'remanent magnetization': [10, 25, 40],
                                'principal susceptibilities': [0.562, 0.85,
                                                               0.2],
                                'susceptibility angles': [90, 0]})
    with raises(AssertionError):
        e.susceptibility_tensor


def test_prolate_ellipsoid_principal_susceptibilities_signal():
    'principal susceptibilities must be all positive'
    e = ProlateEllipsoid(x=1, y=2, z=3, large_axis=6, small_axis=4,
                         strike=10, dip=20, rake=30,
                         props={'remanent magnetization': [10, 25, 40],
                                'principal susceptibilities': [0.562, 0.485,
                                                               -0.2],
                                'susceptibility angles': [90, 0]})
    with raises(AssertionError):
        e.susceptibility_tensor


def test_oblate_ellipsoid_copy():
    'Check the elements of a duplicated ellipsoid'
    orig = OblateEllipsoid(1, 2, 3, 4, 6, 10, 20, 30, {
        'remanent magnetization': [3, -2, 40],
        'susceptibility tensor': [0.562, 0.485, 0.25,
                                  90, 34, 0]})
    cp = orig.copy()
    assert orig is not cp
    assert orig.x == cp.x
    assert orig.y == cp.y
    assert orig.z == cp.z
    assert orig.small_axis == cp.small_axis
    assert orig.large_axis == cp.large_axis
    assert orig.props == cp.props

    cp.x = 4
    cp.y = 6
    cp.z = 7
    cp.props['remanent magnetization'] = [100, -40, -25]
    assert orig.x != cp.x
    assert orig.y != cp.y
    assert orig.z != cp.z
    assert orig.props['remanent magnetization'] != \
        cp.props['remanent magnetization']


def test_oblate_ellipsoid_axes():
    'axes must be given in ascending order'
    raises(AssertionError, OblateEllipsoid, x=1, y=2, z=3,
           small_axis=12, large_axis=4, strike=10, dip=20, rake=30)


def test_oblate_ellipsoid_principal_susceptibilities_fmt():
    'principal susceptibilities must be a list containing 3 elements'
    e = OblateEllipsoid(x=1, y=2, z=3,
                        small_axis=4, large_axis=6,
                        strike=10, dip=20, rake=30,
                        props={'remanent magnetization': [10, 25, 40],
                               'principal susceptibilities': [0.562, 0.485],
                               'susceptibility angles': [90, 0, 0]})

    with raises(AssertionError):
        e.susceptibility_tensor


def test_oblate_ellipsoid_susceptibility_angles_fmt():
    'susceptibility angles must be a list containing 3 elements'
    e = OblateEllipsoid(x=1, y=2, z=3,
                        small_axis=4, large_axis=6,
                        strike=10, dip=20, rake=30,
                        props={'remanent magnetization': [10, 25, 40],
                               'principal susceptibilities': [0.562, 0.485,
                                                              0.2],
                               'susceptibility angles': [90, 0]})

    with raises(AssertionError):
        e.susceptibility_tensor


def test_oblate_ellipsoid_susceptibility_tensor_symm():
    'susceptibility tensor must be symmetric'
    e = OblateEllipsoid(x=1, y=2, z=3,
                        small_axis=4, large_axis=6,
                        strike=10, dip=20, rake=30,
                        props={'remanent magnetization': [10, 25, 40],
                               'principal susceptibilities': [0.562, 0.485,
                                                              0.25],
                               'susceptibility angles': [-240, 71, -2]})
    assert_almost_equal(e.susceptibility_tensor, e.susceptibility_tensor.T,
                        decimal=15)


def test_oblate_ellipsoid_principal_susceptibilities_order():
    'principal susceptibilities must be given in descending order'
    e = OblateEllipsoid(x=1, y=2, z=3, small_axis=4, large_axis=6,
                        strike=10, dip=20, rake=30,
                        props={'remanent magnetization': [10, 25, 40],
                               'principal susceptibilities': [0.562, 0.485,
                                                              0.9],
                               'susceptibility angles': [19, -14, 100]})
    with raises(AssertionError):
        e.susceptibility_tensor


def test_oblate_ellipsoid_principal_susceptibilities_signal():
    'principal susceptibilities must be all positive'
    e = OblateEllipsoid(x=1, y=2, z=3, small_axis=4, large_axis=6,
                        strike=10, dip=20, rake=30,
                        props={'remanent magnetization': [10, 25, 40],
                               'principal susceptibilities': [0.562, 0.485,
                                                              0.9],
                               'susceptibility angles': [19, -14, 100]})
    with raises(AssertionError):
        e.susceptibility_tensor


def test_coord_transf_matrix_oblate_known():
    'Coordinate transformation matrix built with known orientation angles'
    alpha = -np.pi
    gamma = np.pi/2
    delta = 0
    transf_matrix = _coord_transf_matrix_oblate(alpha, gamma, delta)
    assert_almost_equal(transf_matrix, np.identity(3)[[1, 2, 0]], decimal=15)


def test_coord_transf_matrix_oblate_orthonal():
    'Coordinate transformation matrix must be orthogonal'
    alpha = 7
    gamma = 23
    delta = -np.pi/3
    transf_matrix = _coord_transf_matrix_oblate(alpha, gamma, delta)
    dot1 = np.dot(transf_matrix, transf_matrix.T)
    dot2 = np.dot(transf_matrix.T, transf_matrix)
    assert_almost_equal(dot1, dot2, decimal=15)
    assert_almost_equal(dot1, np.identity(3), decimal=15)
    assert_almost_equal(dot2, np.identity(3), decimal=15)


def test__auxiliary_angles_known():
    'Calculate the auxiliary angles with specific input'
    alpha_ref = np.pi - np.arccos(2/np.sqrt(7))
    gamma_ref = np.arctan(2*np.sqrt(1.5))
    delta_ref = np.arcsin(np.sqrt(2)/4)
    strike = 180
    dip = 30
    rake = 45
    alpha, gamma, delta = _auxiliary_angles(strike, dip, rake)
    assert_almost_equal(alpha_ref, alpha, decimal=15)
    assert_almost_equal(gamma_ref, gamma, decimal=15)
    assert_almost_equal(delta_ref, delta, decimal=15)


def test_geometric_element_copy():
    "test that copying yields a different object"
    orig = GeometricElement({'density': 5000})
    cp = orig.copy()
    assert orig.props == cp.props
    assert orig is not cp
    orig.props['density'] = 3000
    assert orig.props != cp.props


def test_polygon_copy():
    "test that copying yields a different object"
    pol = Polygon([[0, 0], [1, 4], [2, 5]], {'density': 500})
    cp = pol.copy()
    assert pol.nverts == cp.nverts
    assert pol.vertices.all() == cp.vertices.all()
    assert pol is not cp


def test_polygonal_prism_copy():
    "test that copying yields a different object"
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


def test_prism_copy():
    "test that copying yields a different object"
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


def test_sphere_copy():
    "test that copying yields a different object"
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


def test_square_copy():
    "test that copying yields a different object"
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


def test_tesseroid_copy():
    "test that copying yields a different object"
    t = Tesseroid(1, 2, 3, 4, 6, 5, {'density': 200})
    cp = t.copy()
    assert t.get_bounds() == cp.get_bounds()
    assert t.props == cp.props
    t.props['density'] = 599
    assert t.props['density'] != cp.props['density']
    assert t is not cp
