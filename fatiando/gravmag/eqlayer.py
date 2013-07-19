"""
Equivalent layer processing.

Use functions here to estimate an equivalent layer from potential field data.
Then you can use the estimated layer to perform tranformations (gridding,
continuation, derivation, reduction to the pole, etc.) by forward modeling
the layer. Use :mod:`fatiando.gravmag.sphere` for forward modeling.

**Functions**

* :func:`~fatiando.gravmag.eqlayer.classic`: The classic equivalent layer with
  damping regularization formulated in the data space
* :func:`~fatiando.gravmag.eqlayer.pel`: The polynomial equivalent layer of
  Oliveira Jr et al. (2012). A more efficient and robust algorithm.

**Data containers**

All equivalent layer functions require that you supply the data in containers.
These are classes that group the data, position arrays, etc.

* :class:`~fatiando.gravmag.eqlayer.TotalField`: Total field magnetic anomaly
* :class:`~fatiando.gravmag.eqlayer.Gz`: Vertical component of gravity (i.e.,
  the gravity anomaly)

**References**

Oliveira Jr., V. C., V. C. F. Barbosa, and L. Uieda (2012), Polynomial
equivalent layer, Geophysics, 78(1), G1-G13, doi:10.1190/geo2012-0196.1.

----

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
    A container for data of the gravity anomaly.

    Coordinate system used: x->North y->East z->Down

    Parameters:

    * x, y, z : 1D arrays
        Arrays with the x, y, z coordinates of the data points

    * data : 1D array
        The values of the data at the observation points

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
    A container for data of the total field magnetic anomaly.

    Coordinate system used: x->North y->East z->Down

    Parameters:

    * x, y, z : 1D arrays
        Arrays with the x, y, z coordinates of the data points

    * data : 1D array
        The values of the data at the observation points

    * inc, dec : floats
        The inclination and declination of the inducing field

    * sinc, sdec : None or floats
        The inclination and declination of the equivalent layer. Use these if
        there is remanent magnetization and the total magnetization of the layer
        if different from the induced magnetization. If there is only induced
        magnetization, use None for sinc and sdec

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

def classic(data, layer, damping=0.):
    """
    The classic equivalent layer in the data space with damping regularization.

    Parameters:

    * data : list
        List of observed data wrapped in data containers (like
        :class:`~fatiando.gravmag.eqlayer.TotalField`). Will make a layer that
        fits all of the observed data. For the moment, still can't mix gravity
        and magnetic data.

    * layer : :class:`fatiando.mesher.PointGrid`
        The equivalent layer

    * damping : float
        The ammount of damping regularization to apply. Must be positive!
        Need to apply enough for the data fit to not be perfect but reflect the
        error in the data.

    Returns:

    * [estimate, predicted] : array, list of arrays
        *estimate* is the estimated physical property distribution. *predicted*
        is a list of the predicted data vector in the same order as supplied in
        *data*

    """
    ndata = sum(d.size for d in data)
    sensitivity = numpy.empty((ndata, layer.size), dtype=float)
    datavec = numpy.empty(ndata, dtype=float)
    bottom = 0
    for d in data:
        sensitivity[bottom:bottom + d.size, :] = d.sensitivity(layer)
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

def pel(data, layer, windows, degree=1, damping=0., smoothness=0.,
        matrices=None):
    """
    The polynomial equivalent layer.

    Parameters:

    * data : list
        List of observed data wrapped in data containers (like
        :class:`~fatiando.gravmag.eqlayer.TotalField`). Will make a layer that
        fits all of the observed data. For the moment, still can't mix gravity
        and magnetic data.

    * layer : :class:`fatiando.mesher.PointGrid`
        The equivalent layer

    * windows : tuple = (ny, nx)
        The number of windows that the layer will be divided in the y and x
        directions

    * degree : int
        The degree of the bivariate polynomials used in each window

    * damping : float
        The amount of damping regularization to apply to the polynomail
        coefficients. Must be positive! A small value (10^-15) usually is
        enough

    * smoothness : float
        How much smoothness regularization to apply to the sources connecting
        adjacent windows. Use this to keep the layer continuous and avoid
        cracks

    * matrices : None or list of 3 arrays
        Use this to supply the matrices generated in a previous run with the
        same data, layer, windows and degree. This way trying different
        regularization parameters doesn't take so long. WARNING: if any other
        parameter changed, the results will be meaningless

    Returns:

    * [estimate, matrices] : array, list of arrays
        *estimate* is the estimated physical property distribution. *matrices*
        is a list of the matrices used to compute the layer.

    """
    ny, nx = windows
    grids = layer.split(windows)
    if layer.shape[1]%nx != 0 or layer.shape[0]%ny != 0:
        raise ValueError(
            'PEL requires windows to be divisable by the grid shape')
    ngrids = len(grids)
    pergrid = _ncoefficients(degree)
    ncoefs = ngrids*pergrid
    if matrices is None:
        modelmatrix, smoothmatrix, rightside = _pel_matrices(data, windows,
            layer, grids, degree)
    else:
        modelmatrix, smoothmatrix, rightside = matrices
    fg = numpy.trace(modelmatrix)
    fr = numpy.trace(smoothmatrix)
    leftside = modelmatrix + (float(smoothness*fg)/fr)*smoothmatrix
    leftside[range(ncoefs),range(ncoefs)] += float(damping*fg)/ncoefs
    coefs = numpy.linalg.solve(leftside, rightside)
    estimate = _coefs2prop(coefs, layer, grids, windows, degree)
    return estimate, [modelmatrix, smoothmatrix, rightside]
