"""
Equivalent layer processing.

Use functions here to estimate an equivalent layer from potential field data.
Then you can use the estimated layer to perform tranformations (gridding,
continuation, derivation, reduction to the pole, etc.) by forward modeling
the layer. Use :mod:`fatiando.gravmag.sphere` for forward modeling.


**References**

Oliveira Jr., V. C., V. C. F. Barbosa, and L. Uieda (2012), Polynomial
equivalent layer, Geophysics, 78(1), G1-G13, doi:10.1190/geo2012-0196.1.

----

"""
from __future__ import division
import numpy
import scipy.sparse

from . import sphere as kernel
from ..utils import dircos, safe_dot
from ..inversion.base import Misfit


class EQLBase(Misfit):
    """
    Base class for the classic equivalent layer.
    """

    def __init__(self, x, y, z, data, grid):
        super(EQLBase, self).__init__(data=data,
            positional={'x':x, 'y':y, 'z':z},
            model={'grid':grid},
            nparams=grid.size, islinear=True)

    def _get_predicted(self, p):
        return safe_dot(self.jacobian(p), p)

class EQLGravity(EQLBase):
    """
    Estimate an equivalent layer from gravity data.

    .. note:: Assumes x = North, y = East, z = Down.

    Parameters:

    * x, y, z : 1d-arrays
        The x, y, z coordinates of each data point.
    * data : 1d-array
        The gravity data at each point.
    * grid : :class:`~fatiando.mesher.PointGrid`
        The sources in the equivalent layer. Will invert for the density of
        each point in the grid.
    * field : string
        Which gravitational field is the data. Options are: ``'gz'`` (gravity
        anomaly), ``'gxx'``, ``'gxy'``, ..., ``'gzz'`` (gravity gradient
        tensor). Defaults to ``'gz'``.

    Examples:

    Use the layer to fit some gravity data and check is our layer is able to
    produce data at a different locations (i.e., interpolate, upward continue)

    >>> import numpy as np
    >>> from fatiando import gridder
    >>> from fatiando.gravmag import sphere, prism
    >>> from fatiando.mesher import Sphere, Prism, PointGrid
    >>> from fatiando.inversion.regularization import Damping
    >>> # Produce some gravity data
    >>> area = (0, 10000, 0, 10000)
    >>> x, y, z = gridder.scatter(area, 500, z=-1, seed=0)
    >>> model = [Prism(4500, 5500, 4500, 5500, 200, 5000, {'density':1000})]
    >>> gz = prism.gz(x, y, z, model)
    >>> # Setup a layer
    >>> layer = PointGrid(area, 500, (25, 25))
    >>> solver = (EQLGravity(x, y, z, gz, layer) +
    ...           10**-26*Damping(layer.size)).fit()
    >>> # Check the fit
    >>> np.allclose(gz, solver.predicted(), rtol=0.01, atol=0.5)
    True
    >>> # Add the densities to the layer
    >>> layer.addprop('density', solver.estimate_)
    >>> # Make a regular grid
    >>> x, y, z = gridder.regular(area, (30, 30), z=-1)
    >>> # Interpolate and check agains the model
    >>> gz_layer = sphere.gz(x, y, z, layer)
    >>> gz_model = prism.gz(x, y, z, model)
    >>> np.allclose(gz_layer, gz_model, rtol=0.01, atol=0.5)
    True
    >>> # Upward continue and check agains model data
    >>> zup = z - 50
    >>> gz_layer = sphere.gz(x, y, zup, layer)
    >>> gz_model = prism.gz(x, y, zup, model)
    >>> np.allclose(gz_layer, gz_model, rtol=0.01, atol=0.5)
    True

    If you have multiple types of gravity data (like gravity anomaly and
    gradient tensor components), you can add EQLGravity instances together for
    a joint inversion:

    >>> x, y, z = gridder.scatter(area, 500, z=-150, seed=0)
    >>> gz = prism.gz(x, y, z, model)
    >>> gzz = prism.gzz(x, y, z, model)
    >>> # Setup a layer
    >>> layer = PointGrid(area, 500, (25, 25))
    >>> solver = (EQLGravity(x, y, z, gz, layer, field='gz') +
    ...           EQLGravity(x, y, z, gzz, layer, field='gzz') +
    ...           10**-24*Damping(layer.size)).fit()
    >>> # Check the fit
    >>> gz_pred, gzz_pred = solver.predicted()
    >>> np.allclose(gz, gz_pred, rtol=0.01, atol=0.5)
    True
    >>> np.allclose(gzz, gzz_pred, rtol=0.01, atol=1)
    True
    >>> # Add the densities to the layer
    >>> layer.addprop('density', solver.estimate_)
    >>> # Upward continue gzz only and check agains model data
    >>> zup = z - 50
    >>> gzz_layer = sphere.gzz(x, y, zup, layer)
    >>> gzz_model = prism.gzz(x, y, zup, model)
    >>> np.allclose(gzz_layer, gzz_model, rtol=0.01, atol=1)
    True

    """

    def __init__(self, x, y, z, data, grid, field='gz'):
        super(EQLGravity, self).__init__(x, y, z, data, grid)
        self.field = field

    def _get_jacobian(self, p):
        x = self.positional['x']
        y = self.positional['y']
        z = self.positional['z']
        func = getattr(kernel, self.field)
        jac = numpy.empty((self.ndata, self.nparams), dtype=float)
        for i, c in enumerate(self.model['grid']):
            jac[:, i] = func(x, y, z, [c], dens=1.)
        return jac

class EQLTotalField(EQLBase):
    """
    Estimate an equivalent layer from total field magnetic anomaly data.

    .. note:: Assumes x = North, y = East, z = Down.

    Parameters:

    * x, y, z : 1d-arrays
        The x, y, z coordinates of each data point.
    * data : 1d-array
        The total field anomaly data at each point.
    * inc, dec : floats
        The inclination and declination of the inducing field
    * grid : :class:`~fatiando.mesher.PointGrid`
        The sources in the equivalent layer. Will invert for the magnetization
        intensity of each point in the grid.
    * sinc, sdec : None or floats
        The inclination and declination of the equivalent layer. Use these if
        there is remanent magnetization and the total magnetization of the
        layer if different from the induced magnetization.
        If there is only induced magnetization, use None

    Examples:

    Use the layer to fit some synthetic data and check is our layer is able to
    produce data at a different locations (i.e., interpolate, upward continue,
    reduce to the pole)


    >>> import numpy as np
    >>> from fatiando import gridder
    >>> from fatiando.gravmag import sphere, prism
    >>> from fatiando.mesher import Sphere, Prism, PointGrid
    >>> from fatiando.inversion.regularization import Damping
    >>> # Produce some synthetic data
    >>> area = (0, 1000, 0, 1000)
    >>> x, y, z = gridder.scatter(area, 500, z=-1, seed=0)
    >>> model = [Prism(450, 550, 450, 550, 100, 500, {'magnetization':5})]
    >>> inc, dec = 10, 23
    >>> tf = prism.tf(x, y, z, model, inc, dec)
    >>> # Setup a layer
    >>> layer = PointGrid(area, 200, (25, 25))
    >>> solver = (EQLTotalField(x, y, z, tf, inc, dec, layer) +
    ...           10**-19*Damping(layer.size)).fit()
    >>> # Check the fit
    >>> np.allclose(tf, solver.predicted(), rtol=0.01, atol=0.5)
    True
    >>> # Add the magnetization to the layer
    >>> layer.addprop('magnetization', solver.estimate_)
    >>> # Make a regular grid
    >>> x, y, z = gridder.regular(area, (30, 30), z=-1)
    >>> # Interpolate and check agains the model
    >>> tf_layer = sphere.tf(x, y, z, layer, inc, dec)
    >>> tf_model = prism.tf(x, y, z, model, inc, dec)
    >>> np.allclose(tf_layer, tf_model, rtol=0.01, atol=0.5)
    True
    >>> # Upward continue and check agains model data
    >>> zup = z - 50
    >>> tf_layer = sphere.tf(x, y, zup, layer, inc, dec)
    >>> tf_model = prism.tf(x, y, zup, model, inc, dec)
    >>> np.allclose(tf_layer, tf_model, rtol=0.01, atol=0.5)
    True
    >>> # Reduce to the pole and check agains model data
    >>> tf_layer = sphere.tf(x, y, zup, layer, 90, 0)
    >>> tf_model = prism.tf(x, y, zup, model, 90, 0)
    >>> np.allclose(tf_layer, tf_model, rtol=0.01, atol=2)
    True

    """

    def __init__(self, x, y, z, data, inc, dec, grid, sinc=None, sdec=None):
        super(EQLTotalField, self).__init__(x, y, z, data, grid)
        self.inc, self.dec = inc, dec
        self.model['inc'] = sinc if sinc is not None else inc
        self.model['dec'] = sdec if sdec is not None else dec

    def _get_jacobian(self, p):
        x = self.positional['x']
        y = self.positional['y']
        z = self.positional['z']
        inc, dec = self.inc, self.dec
        mag = dircos(self.model['inc'], self.model['dec'])
        jac = numpy.empty((self.ndata, self.nparams), dtype=float)
        for i, c in enumerate(self.model['grid']):
            jac[:, i] = kernel.tf(x, y, z, [c], inc, dec, pmag=mag)
        return jac

class PELBase(Misfit):
    """
    Base class for the Polynomial Equivalent Layer.

    .. note::

        Overloads *fit* to convert the estimated coefficients to physical
        properties. The coefficients are stored in the ``coeffs_`` attribute.

    """

    def __init__(self, x, y, z, data, grid, windows, degree):
        super(PELBase, self).__init__(data=data,
            positional={'x':x, 'y':y, 'z':z},
            model={'grid':grid, 'windows':windows, 'degree':degree},
            nparams=windows[0]*windows[1]*ncoeffs(degree),
            islinear=True)

    def _get_predicted(self, p):
        return safe_dot(self.jacobian(p), p)

    def fit(self):
        """
        Solve for the physical property distribution that fits the data.

        Uses the optimization method and parameters defined using the
        :meth:`~fatiando.inversion.base.FitMixin.config` method.

        The estimated physical properties can be accessed through
        :meth:`~fatiando.inversion.base.FitMixin.estimate_`.
        The estimate polynomial coefficients are stored in the ``coeffs_``
        attribute.

        """
        super(self.__class__, self).fit()
        coefs = self.p_
        ny, nx = self.model['windows']
        pergrid = ncoeffs(self.model['degree'])
        estimate = numpy.empty(self.model['grid'].shape, dtype=float)
        grids = self.model['grid'].split(self.model['windows'])
        k = 0
        ystart = 0
        gny, gnx = grids[0].shape
        for i in xrange(ny):
            yend = ystart + gny
            xstart = 0
            for j in xrange(nx):
                xend = xstart + gnx
                g = grids[k]
                estimate[ystart:yend, xstart:xend] = safe_dot(
                        self._bkmatrix(g, degree),
                        coefs[k*pergrid:(k + 1)*pergrid]
                        ).reshape(g.shape)
                xstart = xend
                k += 1
            ystart = yend
        self.coeffs_ = self.p_
        self._estimate = estimate.ravel()
        return self

class PELGravity(PELBase):
    """
    Estimate a polynomial equivalent layer from gravity data.

    .. note:: Assumes x = North, y = East, z = Down.

    Parameters:

    * x, y, z : 1d-arrays
        The x, y, z coordinates of each data point.
    * data : 1d-array
        The gravity data at each point.
    * grid : :class:`~fatiando.mesher.PointGrid`
        The sources in the equivalent layer. Will invert for the density of
        each point in the grid.
    * windows : tuple = (ny, nx)
        The number of windows that the layer will be divided in the y and x
        directions, respectively
    * degree : int
        The degree of the bivariate polynomials used in each window of the PEL
    * field : string
        Which gravitational field is the data. Options are: ``'gz'`` (gravity
        anomaly), ``'gxx'``, ``'gxy'``, ..., ``'gzz'`` (gravity gradient
        tensor). Defaults to ``'gz'``.

    Examples:

    Use the layer to fit some gravity data and check is our layer is able to
    produce data at a different locations (i.e., interpolate, upward continue)

    >>> import numpy as np
    >>> from fatiando import gridder
    >>> from fatiando.gravmag import sphere, prism
    >>> from fatiando.mesher import Sphere, Prism, PointGrid
    >>> # Produce some gravity data
    >>> area = (0, 10000, 0, 10000)
    >>> x, y, z = gridder.scatter(area, 500, z=-1, seed=0)
    >>> model = [Prism(4500, 5500, 4500, 5500, 200, 5000, {'density':1000})]
    >>> gz = prism.gz(x, y, z, model)
    >>> # Setup a layer
    >>> layer = PointGrid(area, 500, (25, 25))
    >>> windows = (20, 20)
    >>> solver = (PELGravity(x, y, z, gz, layer, windows, 1) +
    ...           10**-2*PELSmoothness(layer, windows)).fit()
    >>> # Check the fit
    >>> np.allclose(gz, solver.predicted(), rtol=0.01, atol=0.5)
    True
    >>> # Add the densities to the layer
    >>> layer.addprop('density', solver.estimate_)
    >>> # Make a regular grid
    >>> x, y, z = gridder.regular(area, (30, 30), z=-1)
    >>> # Interpolate and check agains the model
    >>> gz_layer = sphere.gz(x, y, z, layer)
    >>> gz_model = prism.gz(x, y, z, model)
    >>> np.allclose(gz_layer, gz_model, rtol=0.01, atol=0.5)
    True
    >>> # Upward continue and check agains model data
    >>> zup = z - 50
    >>> gz_layer = sphere.gz(x, y, zup, layer)
    >>> gz_model = prism.gz(x, y, zup, model)
    >>> np.allclose(gz_layer, gz_model, rtol=0.01, atol=0.5)
    True

    """

    def __init__(self, x, y, z, data, grid, windows, degree, field='gz'):
        super(PELGravity, self).__init__(x, y, z, data, grid, windows, degre)
        self.field = field

    def _get_jacobian(self, p):
        x = self.positional['x']
        y = self.positional['y']
        z = self.positional['z']
        func = getattr(kernel, self.field)
        jac = numpy.empty((self.ndata, self.nparams), dtype=float)
        for i, c in enumerate(self.model['grid']):
            jac[:,i] = func(x, y, z, [c], dens=1.)
        return jac

def ncoeffs(degree):
    """
    Calculate the number of coefficients in a bivarite polynomail.

    Parameters:

    * degree : int
        The degree of the polynomial

    Returns:

    * n : int
        The number of coefficients

    Examples:

    >>> ncoeffs(1)
    3
    >>> ncoeffs(2)
    6
    >>> ncoeffs(3)
    10
    >>> ncoeffs(4)
    15

    """
    return sum(xrange(1, degree + 2))

#def _bkmatrix(grid, degree):
    #"""
    #Make the Bk polynomial matrix for a PointGrid.

    #>>> from fatiando.mesher import PointGrid
    #>>> grid = PointGrid((0, 1, 0, 2), 10, (2, 2))
    #>>> print _bkmatrix(grid, 2)
    #[[ 1.  0.  0.  0.  0.  0.]
     #[ 1.  0.  1.  0.  0.  1.]
     #[ 1.  2.  0.  4.  0.  0.]
     #[ 1.  2.  1.  4.  2.  1.]]
    #>>> print _bkmatrix(grid, 1)
    #[[ 1.  0.  0.]
     #[ 1.  0.  1.]
     #[ 1.  2.  0.]
     #[ 1.  2.  1.]]
    #>>> print _bkmatrix(grid, 3)
    #[[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     #[ 1.  0.  1.  0.  0.  1.  0.  0.  0.  1.]
     #[ 1.  2.  0.  4.  0.  0.  8.  0.  0.  0.]
     #[ 1.  2.  1.  4.  2.  1.  8.  4.  2.  1.]]

    #"""
    #bmatrix = numpy.transpose(
            #[(grid.x**i)*(grid.y**j)
            #for l in xrange(1, degree + 2)
                #for i, j in zip(xrange(l), xrange(l - 1, -1, -1))])
    #return bmatrix

#def _gkmatrix(data, ndata, grid):
    #"""
    #Make the sensitivity matrix of a subgrid.

    #>>> from fatiando.mesher import PointGrid
    #>>> x, y, z, g = numpy.zeros((4, 500))
    #>>> data = [Gz(x, y, z, g)]
    #>>> grid = PointGrid((0, 10, 0, 10), 10, (10, 10))
    #>>> grid.size
    #100
    #>>> gk = _gkmatrix(data, 500, grid)
    #>>> gk.shape
    #(500, 100)

    #"""
    #sensitivity = numpy.empty((ndata, grid.size), dtype=float)
    #start = 0
    #for d in data:
        #end = start + d.size
        #sensitivity[start:end,:] = d.sensitivity(grid)
        #start = end
    #return sensitivity

#def _rightside(gb, data, ncoefs):
    #vector = numpy.zeros(ncoefs, dtype=float)
    #start = 0
    #for d in data:
        #end = start + d.size
        #vector += numpy.dot(gb[start:end,:].T, d.data)
        #start = end
    #return vector

#def _pel_rmatrix(windows, grid, grids):
    #ny, nx = windows
    #gsize = grids[0].size
    #gny, gnx = grids[0].shape
    #nderivs = (nx - 1)*grid.shape[0] + (ny - 1)*grid.shape[1]
    #rmatrix = lil_matrix((nderivs, grid.size))
    #deriv = 0
    ## derivatives in x
    #for k in xrange(0, len(grids) - ny):
        #bottom = k*gsize + gny*(gnx - 1)
        #top = (k + ny)*gsize
        #for i in xrange(gny):
            #rmatrix[deriv,bottom + i] = -1.
            #rmatrix[deriv,top + 1] = 1.
            #deriv += 1
    ## derivatives in y
    #for k in xrange(0, len(grids)):
        #if (k + 1)%ny == 0:
            #continue
        #right = k*gsize + gny - 1
        #left = (k + 1)*gsize
        #for i in xrange(gnx):
            #rmatrix[deriv,right + i*gny] = -1.
            #rmatrix[deriv,left + i*gny] = 1.
            #deriv += 1
    #return csc_matrix(rmatrix), nderivs

#def _pel_matrices(data, windows, grid, grids, degree):
    #"""
    #Compute the matrices needed by the PEL.

    #1. The Hessian of the model (B^TG^TGB)
    #2. The Hessian of smoothness (B^TR^TRB)
    #3. The right-side vector (B^TG^Td)

    #>>> from fatiando.mesher import PointGrid
    #>>> x, y, z, g = numpy.zeros((4, 500))
    #>>> data = [Gz(x, y, z, g)]
    #>>> grid = PointGrid((0, 10, 0, 10), 10, (10, 10))
    #>>> grid.size
    #100
    #>>> grids = grid.split((2, 2))
    #>>> print [g.size for g in grids]
    #[25, 25, 25, 25]
    #>>> model, smooth, right = _pel_matrices(data, (2, 2), grid, grids, 2)
    #>>> coefs = _ncoefficients(2)*len(grids)
    #>>> coefs
    #24
    #>>> model.shape
    #(24, 24)
    #>>> right.shape
    #(24,)

    #"""
    #ngrids = len(grids)
    #pergrid = _ncoefficients(degree)
    #ncoefs = ngrids*pergrid
    #ndata = sum(d.size for d in data)
    ## make the finite differences matrix for the window borders
    #rmatrix, nderivs = _pel_rmatrix(windows, grid, grids)
    #rightside = numpy.empty(ncoefs, dtype=float)
    #gb = numpy.empty((ndata, ncoefs), dtype=float)
    #rb = numpy.empty((nderivs, ncoefs), dtype=float)
    #st = 0
    #for i, grid in enumerate(grids):
        #bk = _bkmatrix(grid, degree)
        #gk = _gkmatrix(data, ndata, grid)
        #gkbk = numpy.dot(gk, bk)
        #gb[:,i*pergrid:(i + 1)*pergrid] = gkbk
        ## Make a part of the right-side vector
        #rightside[i*pergrid:(i + 1)*pergrid] = _rightside(gkbk, data, pergrid)
        ## Make the RB matrix
        #en = st + grid.size
        #rb[:,i*pergrid:(i + 1)*pergrid] = rmatrix[:,st:en]*bk
        #st = en
    #modelmatrix = numpy.dot(gb.T, gb)
    #smoothmatrix = numpy.dot(rb.T, rb)
    #return modelmatrix, smoothmatrix, rightside


#def pel(data, layer, windows, degree=1, damping=0., smoothness=0.,
        #matrices=None):
    #"""
    #The polynomial equivalent layer.

    #Parameters:

    #* data : list
        #List of observed data wrapped in data containers (like
        #:class:`~fatiando.gravmag.eqlayer.TotalField`). Will make a layer that
        #fits all of the observed data. For the moment, still can't mix gravity
        #and magnetic data.

    #* layer : :class:`fatiando.mesher.PointGrid`
        #The equivalent layer

    #* windows : tuple = (ny, nx)
        #The number of windows that the layer will be divided in the y and x
        #directions

    #* degree : int
        #The degree of the bivariate polynomials used in each window

    #* damping : float
        #The amount of damping regularization to apply to the polynomail
        #coefficients. Must be positive! A small value (10^-15) usually is
        #enough

    #* smoothness : float
        #How much smoothness regularization to apply to the sources connecting
        #adjacent windows. Use this to keep the layer continuous and avoid
        #cracks

    #* matrices : None or list of 3 arrays
        #Use this to supply the matrices generated in a previous run with the
        #same data, layer, windows and degree. This way trying different
        #regularization parameters doesn't take so long. WARNING: if any other
        #parameter changed, the results will be meaningless

    #Returns:

    #* [estimate, matrices] : array, list of arrays
        #*estimate* is the estimated physical property distribution. *matrices*
        #is a list of the matrices used to compute the layer.

    #"""
    #ny, nx = windows
    #grids = layer.split(windows)
    #if layer.shape[1]%nx != 0 or layer.shape[0]%ny != 0:
        #raise ValueError(
            #'PEL requires windows to be divisable by the grid shape')
    #ngrids = len(grids)
    #pergrid = _ncoefficients(degree)
    #ncoefs = ngrids*pergrid
    #if matrices is None:
        #modelmatrix, smoothmatrix, rightside = _pel_matrices(data, windows,
            #layer, grids, degree)
    #else:
        #modelmatrix, smoothmatrix, rightside = matrices
    #fg = numpy.trace(modelmatrix)
    #fr = numpy.trace(smoothmatrix)
    #leftside = modelmatrix + (float(smoothness*fg)/fr)*smoothmatrix
    #leftside[range(ncoefs),range(ncoefs)] += float(damping*fg)/ncoefs
    #coefs = numpy.linalg.solve(leftside, rightside)
    #estimate = _coefs2prop(coefs, layer, grids, windows, degree)
    #return estimate, [modelmatrix, smoothmatrix, rightside]
