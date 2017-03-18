from __future__ import division, absolute_import
from future.builtins import range

import numpy as np
import numpy.testing as npt
from pytest import raises

from ... import gridder
from ..mesh import PrismMesh, Prism, SquareMesh, PointGrid, TesseroidMesh


def test_pointgrid():
    "Test basic functionality of the class"
    area, z, shape = [0, 10, 2, 6], 200, (2, 3)
    g = PointGrid(area, z, shape)
    assert g.shape == shape
    assert g.size == shape[0]*shape[1]
    centers = [[0, 2, 200],
               [0, 4, 200],
               [0, 6, 200],
               [10, 2, 200],
               [10, 4, 200],
               [10, 6, 200]]
    for point, true_c in zip(g, centers):
        npt.assert_allclose(point.center, true_c)
    for i in range(g.size):
        npt.assert_allclose(g[i].center, centers[i])
    for i in range(-1, -(g.size + 1), -1):
        npt.assert_allclose(g[i].center, centers[i])
    npt.assert_allclose(g.x.reshape(g.shape), [[0, 0, 0], [10, 10, 10]])
    npt.assert_allclose(g.y.reshape(g.shape), [[2, 4, 6], [2, 4, 6]])
    npt.assert_allclose(g.z.reshape(g.shape), z + np.zeros(shape))
    npt.assert_allclose(g.dx, 10)
    npt.assert_allclose(g.dy, 2)


def test_pointgrid_z_array():
    "The class behaves correctly when passed an array for z"
    area, shape = [0, 10, 2, 6], (2, 3)
    z = np.arange(shape[0]*shape[1])
    g = PointGrid(area, z, shape)
    assert g.shape == shape
    assert g.size == shape[0]*shape[1]
    centers = [[0, 2, 0],
               [0, 4, 1],
               [0, 6, 2],
               [10, 2, 3],
               [10, 4, 4],
               [10, 6, 5]]
    for point, true_c in zip(g, centers):
        npt.assert_allclose(point.center, true_c)
    for i in range(g.size):
        npt.assert_allclose(g[i].center, centers[i])
    for i in range(-1, -(g.size + 1), -1):
        npt.assert_allclose(g[i].center, centers[i])
    npt.assert_allclose(g.x.reshape(g.shape), [[0, 0, 0], [10, 10, 10]])
    npt.assert_allclose(g.y.reshape(g.shape), [[2, 4, 6], [2, 4, 6]])
    npt.assert_allclose(g.z.reshape(g.shape), z.reshape(shape))
    npt.assert_allclose(g.dx, 10)
    npt.assert_allclose(g.dy, 2)


def test_split():
    "Test if split gives the correct grids"
    z = np.linspace(0, 1100, 12)
    g = PointGrid((0, 3, 0, 2), z, (4, 3))
    g.addprop('bla', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    grids = g.split((2, 3))
    props = [[1, 4], [2, 5], [3, 6], [7, 10], [8, 11], [9, 12]]
    for grid, true in zip(grids, props):
        npt.assert_allclose(grid.props['bla'], true)
    xs = [[0, 1], [0, 1], [0, 1], [2, 3], [2, 3], [2, 3]]
    for grid, true in zip(grids, xs):
        npt.assert_allclose(grid.x, true)
    ys = [[0, 0], [1, 1], [2, 2], [0, 0], [1, 1], [2, 2]]
    for grid, true in zip(grids, ys):
        npt.assert_allclose(grid.y, true)
    zs = [[0, 300], [100, 400], [200, 500], [600, 900], [700, 1000],
          [800, 1100]]
    for grid, true in zip(grids, zs):
        npt.assert_allclose(grid.z, true)


def test_fails_invalid_index():
    "Indexing should fail for an invalid index"
    area, z, shape = [0, 10, 2, 6], 200, (2, 3)
    g = PointGrid(area, z, shape)
    with raises(IndexError):
        g[-7]
    with raises(IndexError):
        g[-500]
    with raises(IndexError):
        g[6]
    with raises(IndexError):
        g[28752]
    with raises(IndexError):
        g[[1]]
    with raises(IndexError):
        g['1']


def test_fails_split():
    "model.split should fail if split shape is not multiple of model.shape"
    area = [-1000., 1000., -2000., 0.]
    shape = (20, 21)
    xp, yp = gridder.regular(area, shape)
    zp = 100*np.arange(xp.size)
    model = PointGrid(area, zp, shape)
    raises(ValueError, model.split, (6, 21))
    raises(ValueError, model.split, (2, 4))
    raises(ValueError, model.split, (3, 5))


def test_z_split_x():
    "model.split along x vs numpy.vsplit splits the z array correctly"
    area = [-1000., 1000., -2000., 0.]
    shape = (20, 21)
    xp, yp = gridder.regular(area, shape)
    zp = 100*np.arange(xp.size)
    model = PointGrid(area, zp, shape)
    subshape = (2, 1)
    submodels = model.split(subshape)
    temp = np.vsplit(np.reshape(zp, shape), subshape[0])
    diff = []
    for i in range(subshape[0]):
        diff.append(np.all((submodels[i].z - temp[i].ravel()) == 0.))
    assert np.alltrue(diff)


def test_z_split_y():
    "model.split along y vs numpy.hsplit splits the z array correctly"
    area = [-1000., 1000., -2000., 0.]
    shape = (20, 21)
    xp, yp = gridder.regular(area, shape)
    zp = 100*np.arange(xp.size)
    model = PointGrid(area, zp, shape)
    subshape = (1, 3)
    submodels = model.split(subshape)
    temp = np.hsplit(np.reshape(zp, shape), subshape[1])
    diff = []
    for i in range(subshape[1]):
        diff.append(np.all((submodels[i].z - temp[i].ravel()) == 0.))
    assert np.alltrue(diff)


def test_point_grid_copy():
    p1 = PointGrid([0, 10, 2, 6], 200, (2, 3))
    p2 = p1.copy()
    assert p1 is not p2
    p1.addprop('density', 3200)
    p2.addprop('density', 2000)
    assert p1.props['density'] != p2.props['density']


def test_prism_mesh_copy():
    p1 = PrismMesh((0, 1, 0, 2, 0, 3), (1, 2, 2))
    p1.addprop('density', 3200 + np.zeros(p1.size))
    p2 = p1.copy()
    assert p1 is not p2
    assert np.array_equal(p1.props['density'], p2.props['density'])


def test_carvetopo():
    bounds = (0, 1, 0, 1, 0, 2)
    shape = (2, 1, 1)
    topox = [0, 0, 1, 1]
    topoy = [0, 1, 0, 1]
    topoz = [-1, -1, -1, -1]
    # Create reference prism meshs
    p0r = []
    p0r.append(None)
    p0r.append(Prism(0, 1, 0, 1, 1, 2))
    p2r = []
    p2r.append(Prism(0, 1, 0, 1, 0, 1))
    p2r.append(None)
    # Create test prism meshes
    p0 = PrismMesh(bounds, shape)
    p0.carvetopo(topox, topoy, topoz)
    p1 = PrismMesh(bounds, shape)
    p1.carvetopo(topox, topoy, topoz, below=False)
    p2 = PrismMesh(bounds, shape)
    p2.carvetopo(topox, topoy, topoz, below=True)
    # Test p0 and p1 which should be the same
    for pi in [p0, p1]:
        for i, p in enumerate(pi):
            if i == 0:
                assert p is None
            else:
                assert p is not None
                assert np.any(p0r[i].center() == p.center())
    # Test p2
    for i, p in enumerate(p2):
        if i == 1:
            assert p is None
        else:
            assert p is not None
            assert np.any(p2r[i].center() == p.center())


def test_square_mesh_copy():
    mesh = SquareMesh((0, 4, 0, 6), (2, 2))
    mesh.addprop('slowness', 234 + np.zeros(mesh.size))
    cp = mesh.copy()
    assert np.array_equal(mesh.props['slowness'], cp.props['slowness'])
    assert mesh is not cp
    assert mesh.bounds == cp.bounds
    assert mesh.dims == cp.dims
    assert np.array_equal(mesh.get_xs(), cp.get_xs())
    assert np.array_equal(mesh.get_ys(), cp.get_ys())
    cp.addprop('density', 3000 + np.zeros(cp.size))
    assert mesh.props != cp.props


def test_tesseroid_mesh_copy():
    orig = TesseroidMesh((0, 1, 0, 2, 3, 0), (1, 2, 2))
    cp = orig.copy()
    assert cp is not orig
    assert orig.celltype == cp.celltype
    assert orig.bounds == cp.bounds
    assert orig.dump == cp.dump
    orig.addprop('density', 3300 + np.zeros(orig.size))
    cp = orig.copy()
    assert np.array_equal(orig.props['density'], cp.props['density'])
