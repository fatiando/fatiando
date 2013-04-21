"""
Equivalent layer processing.
"""
import numpy
from scipy.sparse import linalg

from fatiando.gravmag import sphere as kernel
from fatiando import utils


class Data(object):
    """
    Wrap data for use in the equivalent layer inversion.
    """

    def __init__(self, x, y, z, data):
        self.x = x
        self.y = y
        self.z = z
        self.data = data
        self.size = len(data)

class Gz(Data):
    """
    Wrap gravity anomaly data to be used in the equivalent layer processing.
    """

    def __init__(self, x, y, z, data):
        Data.__init__(self, x, y, z, data)

    def sensitivity(self, grid):
        x, y, z = self.x, self.y, self.z
        sens = numpy.transpose([kernel.gz(x, y, z, [s], dens=1.) for s in grid])
        return sens

class TotalField(Data):
    """
    Wrap total field magnetic anomaly data to be used in the equivalent layer
    processing.
    """

    def __init__(self, x, y, z, data, inc, dec, sinc=None, sdec=None):
        Data.__init__(self, x, y, z, data)
        self.inc = inc
        self.dec = dec
        if sinc is None:
            self.sinc = inc
        else:
            self.sinc = sinc
        if sdec is None:
            self.sdec = dec
        else:
            self.sdec = sdec

    def sensitivity(self, grid):
        x, y, z = self.x, self.y, self.z
        inc, dec = self.inc, self.dec
        mag = utils.dircos(self.sinc, self.sdec)
        sens = numpy.transpose([kernel.tf(x, y, z, [s], inc, dec, pmag=mag)
            for s in grid])
        return sens

def classic(data, grid, damping=0.):
    """
    The classic equivalent layer with damping regularization.
    """
    sensitivity = numpy.concatenate([d.sensitivity(grid) for d in data])
    datavec = numpy.concatenate([d.data for d in data])
    system = numpy.dot(sensitivity, sensitivity.T)
    if damping != 0.:
        order = len(system)
        scale = float(numpy.trace(system))/order
        diag = range(order)
        system[diag, diag] += damping*scale
    tmp = linalg.cg(system, datavec)[0]
    estimate = numpy.dot(sensitivity.T, tmp)
    pred = numpy.dot(sensitivity, estimate)
    predicted = []
    start = 0
    for d in data:
        predicted.append(pred[start:start + d.size])
        start += d.size
    return estimate, predicted

# Polynomial Equivalent Layer (PEL)

def _ncoeffients(degree):
    """
    Return the number of a coefficients in a bivarite polynomail of a given
    degree.

    >>> _ncoeffients(1)
    3
    >>> _ncoeffients(2)
    6
    >>> _ncoeffients(3)
    10
    >>> _ncoeffients(4)
    15

    """
    return sum(xrange(1, degree + 2))

def _bkmatrix(grid, degree):
    """
    Make the Bk polynomial matrix for a PointGrid.

    >>> from fatiando.mesher import PointGrid
    >>> grid = PointGrid((0, 1, 0, 2), 10, (2, 2))
    >>> print _bkmatrix(grid, 2)
    [[ 1.  0.  0.  0.  0.  0.]
     [ 1.  0.  1.  0.  0.  1.]
     [ 1.  2.  0.  4.  0.  0.]
     [ 1.  2.  1.  4.  2.  1.]]
    >>> print _bkmatrix(grid, 1)
    [[ 1.  0.  0.]
     [ 1.  0.  1.]
     [ 1.  2.  0.]
     [ 1.  2.  1.]]
    >>> print _bkmatrix(grid, 3)
    [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 1.  0.  1.  0.  0.  1.  0.  0.  0.  1.]
     [ 1.  2.  0.  4.  0.  0.  8.  0.  0.  0.]
     [ 1.  2.  1.  4.  2.  1.  8.  4.  2.  1.]]

    """
    bmatrix = numpy.transpose(
            [(grid.x**i)*(grid.y**j)
            for l in xrange(1, degree + 2)
                for i, j in zip(xrange(l), xrange(l - 1, -1, -1))])
    return bmatrix

def _gkmatrix(data, ndata, grid):
    sensitivity = numpy.empty((ndata, grid.size), dtype='f')
    start = 0
    for d in data:
        end = start + d.size
        sensitivity[start:end,:] = d.sensitivity(grid)
        start = end
    return sensitivity

def _rightside(gbmatrix, data, ncoefs):
    vector = numpy.zeros(ncoefs, dtype='f')
    start = 0
    for d in data:
        end = start + d.size
        vector += numpy.dot(gb[start:end,:].T, d.data)
        start = end
    return vector

def _pel_system(data, grids, degree, damping, smoothness):
    ngrids = len(grids)
    pergrid = _ncoefficients(degree)
    ncoefs = ngrids*pergrid
    ndata = sum(d.size for d in data)
    rightside = numpy.empty(ncoefs, dtype='f')
    gbmatrices = []
    for i, grid in enumerate(grids):
        bk = _bkmatrix(grid, degree)
        gb = numpy.dot(_gkmatrix(data, ndata, grid), bk)
        gbmatrices.append(gb)
        # Make a part of the right-side vector
        rightside[i*pergrid:(i + 1)*pergrid] = _rightside(gb, data, pergrid)

def pel(data, grid, windows, degree=1, damping=0., smoothness=0.):
    ny, nx = windows
    grids = grid.split(nx, ny)
    leftside, rightside = _pel_system(data, grids, degree, damping, smoothness)

