"""
SrTomo: Straight-ray 2D travel-time tomography (i.e., does not consider
reflection or refraction)

**Functions**

* :func:`~fatiando.seismic.srtomo.run`: Run the tomography on a given data set
* :func:`~fatiando.seismic.srtomo.slowness2vel`: Safely convert slowness to
  velocity (avoids zero division)

**Examples**

Using simple synthetic data::

    >>> import fatiando as ft
    >>> # One source was recorded at 3 receivers.
    >>> # The medium has 2 velocities: 2 and 5
    >>> model = [ft.mesher.Square([0, 10, 0, 5], {'vp':2}),
    ...          ft.mesher.Square([0, 10, 5, 10], {'vp':5})]
    >>> src = (5, 0)
    >>> srcs = [src, src, src]
    >>> recs = [(0, 0), (5, 10), (10, 0)]
    >>> # Calculate the synthetic travel-times
    >>> ttimes = ft.seismic.ttime2d.straight(model, 'vp', srcs, recs)
    >>> print ttimes
    [ 2.5  3.5  2.5]
    >>> # Run the tomography to calculate the 2 velocities
    >>> mesh = ft.mesher.SquareMesh((0, 10, 0, 10), shape=(2, 1))
    >>> # Run the tomography
    >>> estimate, residuals = ft.seismic.srtomo.run(ttimes, srcs, recs, mesh)
    >>> # Actually returns slowness instead of velocity
    >>> for velocity in slowness2vel(estimate):
    ...     print '%.4f' % (velocity),
    2.0000 5.0000
    >>> for v in residuals:
    ...     print '%.4f' % (v),
    0.0000 0.0000 0.0000

Again, using simple synthetic data but this time use Newton's method to solve::

    >>> import fatiando as ft
    >>> # One source was recorded at 3 receivers.
    >>> # The medium has 2 velocities: 2 and 5
    >>> model = [ft.mesher.Square([0, 10, 0, 5], {'vp':2}),
    ...          ft.mesher.Square([0, 10, 5, 10], {'vp':5})]
    >>> src = (5, 0)
    >>> srcs = [src, src, src]
    >>> recs = [(0, 0), (5, 10), (10, 0)]
    >>> # Calculate the synthetic travel-times
    >>> ttimes = ft.seismic.ttime2d.straight(model, 'vp', srcs, recs)
    >>> print ttimes
    [ 2.5  3.5  2.5]
    >>> # Run the tomography to calculate the 2 velocities
    >>> mesh = ft.mesher.SquareMesh((0, 10, 0, 10), shape=(2, 1))
    >>> # Will use Newton's method to solve this
    >>> solver = ft.inversion.gradient.newton(initial=[0, 0], maxit=5)
    >>> estimate, residuals = ft.seismic.srtomo.run(ttimes, srcs, recs, mesh,
    ...                                          solver)
    >>> # Actually returns slowness instead of velocity
    >>> for velocity in slowness2vel(estimate):
    ...     print '%.4f' % (velocity),
    2.0000 5.0000
    >>> for v in residuals:
    ...     print '%.4f' % (v),
    0.0000 0.0000 0.0000

.. note:: A simple way to plot the results is to use the ``addprop`` method of
    the mesh and then pass the mesh to :func:`fatiando.vis.map.squaremesh`.

----

"""
import numpy
import scipy.sparse

from fatiando import inversion, utils
from fatiando.seismic import ttime2d


class TravelTime(inversion.datamodule.DataModule):
    """
    Data module for the 2D travel-time straight-ray tomography problem.

    Bundles together the travel-time data, source and receiver positions
    and uses this to calculate gradients, Hessians and other inverse problem
    related quantities.

    Parameters:

    * ttimes : array
        Array with the travel-times of the straight seismic rays.
    * srcs : list of lists
        List of the [x, y] positions of the sources.
    * recs : list of lists
        List of the [x, y] positions of the receivers.
    * mesh : :class:`~fatiando.mesher.SquareMesh` or compatible
        The mesh where the inversion (tomography) will take place.
    * sparse : True or False
        Wether or not to use sparse matrices from scipy

    The ith travel-time is the time between the ith element in *srcs* and the
    ith element in *recs*.

    For example::

        >>> from fatiando.mesher import SquareMesh
        >>> # One source
        >>> src = (5, 0)
        >>> # was recorded at 3 receivers
        >>> recs = [(0, 0), (5, 10), (10, 0)]
        >>> # Resulting in Vp travel times
        >>> ttimes = [2.5, 5., 2.5]
        >>> # If running the tomography on a mesh
        >>> mesh = SquareMesh(bounds=(0, 10, 0, 10), shape=(10, 10))
        >>> # what TravelTime expects is
        >>> srcs = [src, src, src]
        >>> dm = TravelTime(ttimes, srcs, recs, mesh)

    """

    def __init__(self, ttimes, srcs, recs, mesh, sparse=False):
        inversion.datamodule.DataModule.__init__(self, ttimes)
        self.srcs = srcs
        self.recs = recs
        self.mesh = mesh
        self.nparams = mesh.size
        self.ndata = len(ttimes)
        self.sparse = sparse
        if sparse:
            self.get_predicted = self._get_predicted_sparse
            self.sum_gradient = self._sum_gradient_sparse
        else:
            self.get_predicted = self._get_predicted
            self.sum_gradient = self._sum_gradient
        self.jacobian = self._get_jacobian()
        self.jacobian_T = self.jacobian.T
        self.hessian = None

    def _get_jacobian(self):
        """
        Build the Jacobian (sensitivity) matrix using the travel-time data
        stored.
        """
        srcs, recs = self.srcs, self.recs
        if not self.sparse:
            jac = numpy.array(
                [ttime2d.straight([cell], '', srcs, recs, velocity=1.)
                 for cell in self.mesh]).T
        else:
            shoot = ttime2d.straight
            nonzero = []
            extend = nonzero.extend
            for j, c in enumerate(self.mesh):
                extend((i, j, tt)
                    for i, tt in enumerate(shoot([c], '', srcs, recs,
                                            velocity=1.))
                    if tt != 0)
            row, col, val = numpy.array(nonzero).T
            shape = (self.ndata, self.nparams)
            jac = scipy.sparse.csr_matrix((val, (row, col)), shape)
        return jac

    def _get_hessian(self):
        """
        Build the Hessian matrix by Gauss-Newton approximation.
        """
        if not self.sparse:
            hess = numpy.dot(self.jacobian_T, self.jacobian)
        else:
            hess = self.jacobian_T*self.jacobian
        return hess

    def _get_predicted(self, p):
        return numpy.dot(self.jacobian, p)

    def _get_predicted_sparse(self, p):
        return self.jacobian*p

    def _sum_gradient(self, gradient, p=None, residuals=None):
        if p is None:
            return gradient - 2.*numpy.dot(self.jacobian_T, self.data)
        else:
            return gradient - 2.*numpy.dot(self.jacobian_T, residuals)

    def _sum_gradient_sparse(self, gradient, p=None, residuals=None):
        if p is None:
            return gradient - 2.*self.jacobian_T*self.data
        else:
            return gradient - 2.*self.jacobian_T*residuals

    def sum_hessian(self, hessian, p=None):
        if self.hessian is None:
            self.hessian = self._get_hessian()
        return hessian + 2.*self.hessian

def slowness2vel(slowness, tol=10**(-8)):
    """
    Safely convert slowness to velocity.

    Almost 0 slowness is mapped to 0 velocity.

    Parameters:

    * slowness : array
        The slowness values
    * tol : float
        Slowness < tol will be set to 0 velocity

    Returns:

    * velocity : array
        The converted velocities

    """
    def safe(s):
        if abs(s) >= tol:
            return 1./s
        else:
            return 0.
    velocity = numpy.fromiter((safe(s) for s in slowness), dtype='f')
    return velocity

def run(ttimes, srcs, recs, mesh, solver=None, sparse=False, damping=0.,
    smooth=0., sharp=0., beta=10.**(-5)):
    """
    Perform a 2D straight-ray travel-time tomography. Estimates the slowness
    (1/velocity) of cells in mesh (because slowness is linear and easier)

    Regularization is usually **not** optional. At least some amount of damping
    is required.

    Parameters:

    * ttimes : array
        The travel-times of the straight seismic rays.
    * srcs : list of lists
        List of the [x, y] positions of the sources.
    * recs : list of lists
        List of the [x, y] positions of the receivers.
    * mesh : :class:`~fatiando.mesher.SquareMesh` or compatible
        The mesh where the inversion (tomography) will take place.
    * solver : function
        A linear or non-linear inverse problem solver generated by a factory
        function from a module of package :mod:`fatiando.inversion`. If None,
        will use the default solver.
    * sparse : True or False
        If True, will use sparse matrices from `scipy.sparse`.

        .. note:: If you provided a solver function, don't forget to turn on
            sparcity in the inversion solver module **BEFORE** creating the
            solver function! The usual way of doing this is by calling the
            ``use_sparse`` function. Ex:
            ``fatiando.inversion.gradient.use_sparse()``

        .. warning:: Jacobian matrix building using sparse matrices isn't very
            optimized. It will be slow but won't overflow the memory.

    * damping : float
        Damping regularizing parameter (i.e., how much damping to apply).
        Must be a positive scalar.
    * smooth : float
        Smoothness regularizing parameter (i.e., how much smoothness to apply).
        Must be a positive scalar.
    * sharp : float
        Sharpness (total variation) regularizing parameter (i.e., how much
        sharpness to apply). Must be a positive scalar.
    * beta : float
        Total variation parameter. See
        :class:`fatiando.inversion.regularizer.TotalVariation` for details

    Returns:

    * results : list = [slowness, residuals]:

        * slowness : array
            The slowness of each cell in *mesh*
        * residuals : array
            The inversion residuals (observed travel-times minus predicted
            travel-times by the slowness estimate)

    """
    if len(ttimes) != len(srcs) != len(recs):
        msg = "Must have same number of travel-times, sources and receivers"
        raise ValueError(msg)
    if damping < 0:
        raise ValueError("Damping must be positive")
    # If no solver is given, generate custom ones
    if solver is None:
        if sharp == 0:
            if sparse:
                inversion.linear.use_sparse()
            solver = inversion.linear.overdet(mesh.size)
        else:
            if sparse:
                inversion.gradient.use_sparse()
            solver = inversion.gradient.levmarq(
                initial=numpy.ones(mesh.size, dtype='f'))
    nparams = len(mesh)
    # Make the data modules and regularizers
    dms = [TravelTime(ttimes, srcs, recs, mesh, sparse)]
    regs = []
    if damping:
        regs.append(inversion.regularizer.Damping(damping, nparams,
            sparse=sparse))
    if smooth:
        regs.append(inversion.regularizer.Smoothness2D(smooth, mesh.shape,
            sparse=sparse))
    if sharp:
        regs.append(inversion.regularizer.TotalVariation2D(sharp, mesh.shape,
            beta, sparse=sparse))
    try:
        for i, chset in enumerate(solver(dms, regs)):
            continue
    except numpy.linalg.linalg.LinAlgError:
        raise ValueError("Oops, the Hessian is a singular matrix." +
                         " Try applying more regularization")
    slowness = chset['estimate']
    residuals = chset['residuals'][0]
    return slowness, residuals
