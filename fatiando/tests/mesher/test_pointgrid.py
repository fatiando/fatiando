from __future__ import division
import numpy as np
import numpy.testing as npt
from pytest import raises
from future.builtins import range

from ...mesher import PointGrid
from ... import gridder


area = [-1000., 1000., -2000., 0.]
shape = (20, 21)
xp, yp = gridder.regular(area, shape)
zp = 100*np.arange(xp.size)
model = PointGrid(area, zp, shape)


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
    with raises(IndexError) as e:
        g[-7]
    with raises(IndexError) as e:
        g[-500]
    with raises(IndexError) as e:
        g[6]
    with raises(IndexError) as e:
        g[28752]
    with raises(IndexError) as e:
        g[[1]]
    with raises(IndexError) as e:
        g['1']


def test_fails_split():
    "model.split should fail if split shape is not multiple of model.shape"
    raises(ValueError, model.split, (6, 21))
    raises(ValueError, model.split, (2, 4))
    raises(ValueError, model.split, (3, 5))


def test_z_split_x():
    "model.split along x vs numpy.vsplit splits the z array correctly"
    subshape = (2, 1)
    submodels = model.split(subshape)
    temp = np.vsplit(np.reshape(zp, shape), subshape[0])
    diff = []
    for i in range(subshape[0]):
        diff.append(np.all((submodels[i].z - temp[i].ravel()) == 0.))
    assert np.alltrue(diff)


def test_z_split_y():
    "model.split along y vs numpy.hsplit splits the z array correctly"
    subshape = (1, 3)
    submodels = model.split(subshape)
    temp = np.hsplit(np.reshape(zp, shape), subshape[1])
    diff = []
    for i in range(subshape[1]):
        diff.append(np.all((submodels[i].z - temp[i].ravel()) == 0.))
    assert np.alltrue(diff)
