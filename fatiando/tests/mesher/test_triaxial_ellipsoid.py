from ...mesher import TriaxialEllipsoid
import numpy as np
from numpy.random import rand


def test_triaxial_ellipsoid_copy():
    orig = TriaxialEllipsoid(1, 2, 3, 6, 5, 4, 10, 20, 30, {
                             'remanence': [10000., 25., 40.],
                             'k': [0.562, 0.485, 0.25, 90., 0., 0.]})
    cp = orig.copy()
    assert orig is not cp
    assert orig.x == cp.x
    assert orig.y == cp.y
    assert orig.z == cp.z
    assert orig.a == cp.a
    assert orig.b == cp.b
    assert orig.c == cp.c
    assert orig.props == cp.props
    assert np.array_equal(orig.V(), cp.V())

    cp.x = 4
    cp.y = 6
    cp.z = 7
    cp.props['k'] = [0.7, 0.9, 10., 90., 28., -10.]
    assert orig.x != cp.x
    assert orig.y != cp.y
    assert orig.z != cp.z
    assert orig.props['k'] != cp.props['k']


def test_triaxial_ellipsoid_V_orthogonal():
    a = 1000.*rand()
    b = 0.8*a
    c = 0.6*a
    pi = np.pi
    e = TriaxialEllipsoid(1000.*rand(), 1000.*rand(), 1000.*rand(),
                          a, b, c,
                          pi*rand(), 0.5*pi*rand(), 0.5*pi*rand())
    assert np.allclose(np.dot(e.V().T, e.V()), np.identity(3))
    assert np.allclose(np.dot(e.V(), e.V().T), np.identity(3))


def test_triaxial_ellipsoid_V_identity():
    a = 1000.*rand()
    b = 0.8*a
    c = 0.6*a
    pi = np.pi
    e = TriaxialEllipsoid(1000.*rand(), 1000.*rand(), 1000.*rand(),
                          a, b, c,
                          180., 90., 0.)
    assert np.allclose(e.V(), np.identity(3))
