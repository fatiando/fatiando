from ...mesher import OblateEllipsoid
import numpy as np
from numpy.random import rand


def test_oblate_ellipsoid_copy():
    orig = OblateEllipsoid(1, 2, 3, 4, 6, 10, 30,
                            {'remanence': [10000., 25., 40.], 
                             'k': [0.562, 0.485, 0.25, 90., 0., 0.]})
    cp = orig.copy()
    assert orig is not cp
    assert orig.x == cp.x
    assert orig.y == cp.y
    assert orig.z == cp.z
    assert orig.a == cp.a
    assert orig.b == cp.b
    assert orig.props == cp.props
    assert np.array_equal(orig.V(), cp.V())

    cp.x = 4
    cp.y = 6
    cp.z = 7
    cp.props['remanence'] = [100., -40., -25.]
    assert orig.x != cp.x
    assert orig.y != cp.y
    assert orig.z != cp.z
    assert orig.props['remanence'] != cp.props['remanence']
    
def test_oblate_ellipsoid_V_orthogonal():
    a = 1000.*rand()
    b = 1.5*a
    pi = np.pi
    e = OblateEllipsoid(1000.*rand(), 1000.*rand(), 1000.*rand(), 
                          a, b, pi*rand(), 0.5*pi*rand())
    assert np.allclose(np.dot(e.V().T,e.V()), np.identity(3))
    assert np.allclose(np.dot(e.V(),e.V().T), np.identity(3))

def test_oblate_ellipsoid_V_specific():
    a = 1000.*rand()
    b = 2.4*a
    pi = np.pi
    e = OblateEllipsoid(1000.*rand(), 1000.*rand(), 1000.*rand(), 
                          a, b, 180., 90.)
    true = np.array([[-1, 0, 0],
                     [ 0, 0,-1],
                     [ 0,-1, 0]])
    assert np.allclose(e.V(),true)