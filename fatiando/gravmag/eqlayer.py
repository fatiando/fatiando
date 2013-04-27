"""
Equivalent layer processing.
"""
import numpy
from scipy.sparse import linalg, lil_matrix, csc_matrix

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
        sens = numpy.empty((self.size, len(grid)), dtype=float)
        for i, s in enumerate(grid):
            sens[:,i] = kernel.gz(x, y, z, [s], dens=1.)
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
        sens = numpy.empty((self.size, len(grid)), dtype=float)
        for i, s in enumerate(grid):
            sens[:,i] = kernel.tf(x, y, z, [s], inc, dec, pmag=mag)
        return sens

def classic(data, grid, damping=0.):
    """
    The classic equivalent layer with damping regularization.
    """
    ndata = sum(d.size for d in data)
    sensitivity = numpy.empty((ndata, grid.size), dtype=float)
    datavec = numpy.empty(ndata, dtype=float)
    bottom = 0
    for d in data:
        sensitivity[bottom:bottom + d.size, :] = d.sensitivity(grid)
        datavec[bottom:bottom + d.size] = d.data
        bottom += d.size
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

def _ncoefficients(degree):
    """
    Return the number of a coefficients in a bivarite polynomail of a given
    degree.

    >>> _ncoefficients(1)
    3
    >>> _ncoefficients(2)
    6
    >>> _ncoefficients(3)
    10
    >>> _ncoefficients(4)
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
    """
    Make the sensitivity matrix of a subgrid.

    >>> from fatiando.mesher import PointGrid
    >>> x, y, z, g = numpy.zeros((4, 500))
    >>> data = [Gz(x, y, z, g)]
    >>> grid = PointGrid((0, 10, 0, 10), 10, (10, 10))
    >>> grid.size
    100
    >>> gk = _gkmatrix(data, 500, grid)
    >>> gk.shape
    (500, 100)

    """
    sensitivity = numpy.empty((ndata, grid.size), dtype=float)
    start = 0
    for d in data:
        end = start + d.size
        sensitivity[start:end,:] = d.sensitivity(grid)
        start = end
    return sensitivity

def _rightside(gb, data, ncoefs):
    vector = numpy.zeros(ncoefs, dtype=float)
    start = 0
    for d in data:
        end = start + d.size
        vector += numpy.dot(gb[start:end,:].T, d.data)
        start = end
    return vector

def _pel_rmatrix(windows, grid, grids):
    ny, nx = windows
    gsize = grids[0].size
    gny, gnx = grids[0].shape
    nderivs = (nx - 1)*grid.shape[0] + (ny - 1)*grid.shape[1]
    rmatrix = lil_matrix((nderivs, grid.size))
    deriv = 0
    # derivatives in x
    for k in xrange(0, len(grids) - ny):
        bottom = k*gsize + gny*(gnx - 1)
        top = (k + ny)*gsize
        for i in xrange(gny):
            rmatrix[deriv,bottom + i] = -1.
            rmatrix[deriv,top + 1] = 1.
            deriv += 1
    # derivatives in y
    for k in xrange(0, len(grids)):
        if (k + 1)%ny == 0:
            continue
        right = k*gsize + gny - 1
        left = (k + 1)*gsize
        for i in xrange(gnx):
            rmatrix[deriv,right + i*gny] = -1.
            rmatrix[deriv,left + i*gny] = 1.
            deriv += 1
    return csc_matrix(rmatrix), nderivs

def _pel_matrices(data, windows, grid, grids, degree):
    """
    Compute the matrices needed by the PEL.

    1. The Hessian of the model (B^TG^TGB)
    2. The Hessian of smoothness (B^TR^TRB)
    3. The right-side vector (B^TG^Td)

    >>> from fatiando.mesher import PointGrid
    >>> x, y, z, g = numpy.zeros((4, 500))
    >>> data = [Gz(x, y, z, g)]
    >>> grid = PointGrid((0, 10, 0, 10), 10, (10, 10))
    >>> grid.size
    100
    >>> grids = grid.split((2, 2))
    >>> print [g.size for g in grids]
    [25, 25, 25, 25]
    >>> model, smooth, right = _pel_matrices(data, (2, 2), grid, grids, 2)
    >>> coefs = _ncoefficients(2)*len(grids)
    >>> coefs
    24
    >>> model.shape
    (24, 24)
    >>> right.shape
    (24,)

    """
    ngrids = len(grids)
    pergrid = _ncoefficients(degree)
    ncoefs = ngrids*pergrid
    ndata = sum(d.size for d in data)
    # make the finite differences matrix for the window borders
    rmatrix, nderivs = _pel_rmatrix(windows, grid, grids)
    rightside = numpy.empty(ncoefs, dtype=float)
    gb = numpy.empty((ndata, ncoefs), dtype=float)
    rb = numpy.empty((nderivs, ncoefs), dtype=float)
    st = 0
    for i, grid in enumerate(grids):
        bk = _bkmatrix(grid, degree)
        gk = _gkmatrix(data, ndata, grid)
        gkbk = numpy.dot(gk, bk)
        gb[:,i*pergrid:(i + 1)*pergrid] = gkbk
        # Make a part of the right-side vector
        rightside[i*pergrid:(i + 1)*pergrid] = _rightside(gkbk, data, pergrid)
        # Make the RB matrix
        en = st + grid.size
        rb[:,i*pergrid:(i + 1)*pergrid] = rmatrix[:,st:en]*bk
        st = en
    modelmatrix = numpy.dot(gb.T, gb)
    smoothmatrix = numpy.dot(rb.T, rb)
    return modelmatrix, smoothmatrix, rightside

def _coefs2prop(coefs, grid, grids, windows, degree):
    """
    Convert the coefficients to the physical property estimate.
    """
    ny, nx = windows
    pergrid = _ncoefficients(degree)
    estimate = numpy.empty(grid.shape, dtype=float)
    k = 0
    ystart = 0
    gny, gnx = grids[0].shape
    for i in xrange(ny):
        yend = ystart + gny
        xstart = 0
        for j in xrange(nx):
            xend = xstart + gnx
            g = grids[k]
            estimate[ystart:yend,xstart:xend] = numpy.dot(_bkmatrix(g, degree),
                coefs[k*pergrid:(k + 1)*pergrid]).reshape(g.shape)
            xstart = xend
            k += 1
        ystart = yend
    return estimate.ravel()

def pel(data, grid, windows, degree=1, damping=0., smoothness=0.,
        matrices=None):
    """
    The Polynomial Equivalent Layers
    """
    ny, nx = windows
    grids = grid.split(windows)
    if grid.shape[1]%nx != 0 or grid.shape[0]%ny != 0:
        raise ValueError(
            'PEL requires windows to be divisable by the grid shape')
    ngrids = len(grids)
    pergrid = _ncoefficients(degree)
    ncoefs = ngrids*pergrid
    if matrices is None:
        modelmatrix, smoothmatrix, rightside = _pel_matrices(data, windows,
            grid, grids, degree)
    else:
        modelmatrix, smoothmatrix, rightside = matrices
    fg = numpy.trace(modelmatrix)
    fr = numpy.trace(smoothmatrix)
    leftside = modelmatrix + (float(smoothness*fg)/fr)*smoothmatrix
    leftside[range(ncoefs),range(ncoefs)] += float(damping*fg)/ncoefs
    coefs = numpy.linalg.solve(leftside, rightside)
    estimate = _coefs2prop(coefs, grid, grids, windows, degree)
    return estimate, [modelmatrix, smoothmatrix, rightside]
