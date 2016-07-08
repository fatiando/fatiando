import numpy as np

from fatiando.mesher import Pointgrid
from fatiando import gridder

model = None
xp, yp, zp = None, None, None


def setup():
    global model, xp, yp, zp, area, shape
    area = [-1000., 1000., -2000., 0.]
    shape = (20, 21)
    xp, yp = gridder.regular(area, shape)
    zp = 100.*np.random.rand(xp.size)
    model = PointGrid(area, zp, shape)


def test_z():
    "model.z gives the original array zp"
    diff = model.z-zp
    assert np.all(diff == 0.)


def test_split_x():
    "model.split along x vs numpy.vsplit"
    subshape = (2, 1)
    submodels = model.split(subshape)
    temp = np.vsplit(np.reshape(zp, shape), subshape[0])
    diff = []
    for i in range(subshape[0]):
        diff.append(np.all((submodels[i].z - temp[i].ravel()) == 0.))
    assert np.alltrue(diff)


def test_split_y():
    "model.split along y vs numpy.hsplit"
    subshape = (1, 3)
    submodels = model.split(subshape)
    temp = np.hsplit(np.reshape(zp, shape), subshape[1])
    diff = []
    for i in range(subshape[1]):
        diff.append(np.all((submodels[i].z - temp[i].ravel()) == 0.))
    assert np.alltrue(diff)
