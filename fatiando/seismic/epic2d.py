"""
Epicenter determination in 2D, i.e., assuming a flat Earth.

**Homogeneous Earth**

* :func:`~fatiando.seismic.epic2d.homogeneous`

Estimates the (x, y) cartesian coordinates of the epicenter based on travel-time
residuals between S and P waves, assuming a homogeneous velocity distribution.

Example using synthetic data::

    >>> import fatiando as ft
    >>> # Generate synthetic travel-time residuals
    >>> area = (0, 10, 0, 10)
    >>> vp = 2
    >>> vs = 1
    >>> model = [ft.msh.dd.Square(area, props={'vp':vp, 'vs':vs})]
    >>> # The true source (epicenter)
    >>> src = (5, 5)
    >>> recs = [(5, 0), (5, 10), (10, 0)]
    >>> srcs = [src, src, src]
    >>> ptime = ft.seis.ttime2d.straight(model, 'vp', srcs, recs)
    >>> stime = ft.seis.ttime2d.straight(model, 'vs', srcs, recs)
    >>> ttres = stime - ptime
    >>> # Solve using Newton's
    >>> # Generate synthetic travel-time residuals method
    >>> solver = ft.inversion.gradient.newton(initial=(1, 1), tol=10**(-3),
    ...     maxit=1000)
    >>> # Estimate the epicenter
    >>> p, residuals = ft.seis.epic2d.homogeneous(ttres, recs, vp, vs, solver)
    >>> print "(%.4f, %.4f)" % (p[0], p[1])
    (5.0000, 5.0000)

Example using ``iterate = True`` to step through the solver algorithm::

    >>> import fatiando as ft
    >>> # Generate synthetic travel-time residuals
    >>> area = (0, 10, 0, 10)
    >>> vp = 2
    >>> vs = 1
    >>> model = [ft.msh.dd.Square(area, props={'vp':vp, 'vs':vs})]
    >>> # The true source (epicenter)
    >>> src = (5, 5)
    >>> recs = [(5, 0), (5, 10), (10, 0)]
    >>> srcs = [src, src, src]
    >>> ptime = ft.seis.ttime2d.straight(model, 'vp', srcs, recs)
    >>> stime = ft.seis.ttime2d.straight(model, 'vs', srcs, recs)
    >>> ttres = stime - ptime
    >>> # Solve using Newton's
    >>> # Generate synthetic travel-time residuals method
    >>> solver = ft.inversion.gradient.newton(initial=(1, 1), tol=10**(-3),
    ...     maxit=5)    
    >>> # Show the steps to estimate the epicenter
    >>> steps = ft.seis.epic2d.homogeneous(ttres, recs, vp, vs, solver,
    ...     iterate=True)
    >>> for p, r in steps:
    ...     print "(%.4f, %.4f)" % (p[0], p[1])
    (1.0000, 1.0000)
    (2.4157, 5.8424)
    (4.3279, 4.7485)
    (4.9465, 4.9998)
    (4.9998, 5.0000)
    (5.0000, 5.0000)
    
----

"""

import time

import numpy

from fatiando import logger, inversion, utils


log = logger.dummy('fatiando.seismic.epic2d')

class TTRFlat(inversion.datamodule.DataModule):
    """
    Data module for epicenter estimation using travel-time residuals between
    S and P waves, assuming a homogeneous Earth.

    The travel-time residual measured by the ith receiver is a function of the
    (x, y) coordinates of the epicenter:

    .. math::

        t_{S_i} - t_{P_i} = \\Delta t_i (x, y) =
        \\left(\\frac{1}{V_S} - \\frac{1}{V_P} \\right)
        \\sqrt{(x_i - x)^2 + (y_i - y)^2}

    The elements :math:`G_{i1}` and :math:`G_{i2}` of the Jacobian matrix for
    this data type are

    .. math::

        G_{i1}(x, y) = -\\left(\\frac{1}{V_S} - \\frac{1}{V_P} \\right)
        \\frac{x_i - x}{\\sqrt{(x_i - x)^2 + (y_i - y)^2}}

    .. math::

        G_{i2}(x, y) = -\\left(\\frac{1}{V_S} - \\frac{1}{V_P} \\right)
        \\frac{y_i - y}{\\sqrt{(x_i - x)^2 + (y_i - y)^2}}
        
    The Hessian matrix is approximated by
    :math:`2\\bar{\\bar{G}}^T\\bar{\\bar{G}}` (Gauss-Newton method).
        
    Parameters:

    * ttres : array
        Travel-time residuals between S and P waves
    * recs : list of lists
        List with the (x, y) coordinates of the receivers
    * vp : float
        Assumed velocity of P waves
    * vs : float
        Assumed velocity of S waves
    
    """

    def __init__(self, ttres, recs, vp, vs):
        inversion.datamodule.DataModule.__init__(self, ttres)
        self.x_rec, self.y_rec = numpy.array(recs, dtype='f').T
        self.vp = vp
        self.vs = vs
        self.alpha = 1./float(vs) - 1./float(vp)

    def get_predicted(self, p):
        x, y = p
        return self.alpha*numpy.sqrt((self.x_rec - x)**2 + (self.y_rec - y)**2)
        
    def sum_gradient(self, gradient, p=None, residuals=None):
        x, y = p
        sqrt = numpy.sqrt((self.x_rec - x)**2 + (self.y_rec - y)**2)
        jac_T = [-self.alpha*(self.x_rec - x)/sqrt,
                 -self.alpha*(self.y_rec - y)/sqrt]
        return gradient - 2.*numpy.dot(jac_T, residuals)

    def sum_hessian(self, hessian, p=None):
        x, y = p
        sqrt = numpy.sqrt((self.x_rec - x)**2 + (self.y_rec - y)**2)
        jac_T = numpy.array([-self.alpha*(self.x_rec - x)/sqrt,
                             -self.alpha*(self.y_rec - y)/sqrt])
        return hessian + 2.*numpy.dot(jac_T, jac_T.T)        

class _MinimumDistance(inversion.regularizer.Regularizer):
    """
    A regularizing function that imposes that the solution (estimated epicenter)
    be at the smallest distance from the receivers as possible.

    .. math::

        \\theta(\\bar{p}) = \\bar{d}^T\\bar{d}

    where :math:`\\bar{d}` is a vector with the distance from the current
    estimate :math:`\\bar{p} = \\begin{bmatrix}x && y\end{bmatrix}^T` to the
    receivers.

    .. math::

        d_i = \\sqrt{(x_i - x)^2 + (y_i - y)^2}

    The gradient vector of :math:`\\theta(\\bar{p})` is given by

    .. math::

        \\bar{g}(\\bar{p}) = -2\\begin{bmatrix}
        \\sum\\limits_{i=1}^N x_i - x \\\\
        \\sum\\limits_{i=1}^N y_i - y \end{bmatrix}
        
    The elements :math:`G_{i1}` and :math:`G_{i2}` of the Jacobian matrix of
    :math:`\\theta(\\bar{p})` are

    .. math::

        G_{i1}(x, y) = -\\frac{x_i - x}{\\sqrt{(x_i - x)^2 + (y_i - y)^2}}

    .. math::

        G_{i2}(x, y) = -\\frac{y_i - y}{\\sqrt{(x_i - x)^2 + (y_i - y)^2}}

    And the Hessian matrix can be approximated by
    :math:`2\\bar{\\bar{G}}^T\\bar{\\bar{G}}`.
        
    """

    def __init__(self, mu, recs):
        inversion.regularizer.Regularizer.__init__(self, mu)
        self.xrec, self.yrec = numpy.array(recs, dtype='f').T   

    def value(self, p):
        x, y = p
        distance = numpy.sqrt((self.xrec - x)**2 + (self.yrec - y)**2)
        return self.mu*numpy.linalg.norm(distance)**2

    def sum_gradient(self, gradient, p):
        x, y = p
        grad = [numpy.sum(self.xrec - x), numpy.sum(self.yrec - y)]
        return gradient + self.mu*-2.*numpy.array(grad)
        
    def sum_hessian(self, hessian, p):
        x, y = p
        sqrt = numpy.sqrt((self.xrec - x)**2 + (self.yrec - y)**2)
        jac_T = numpy.array([(x - self.xrec)/sqrt,
                             (y - self.yrec)/sqrt])
        return hessian + self.mu*2.*numpy.dot(jac_T, jac_T.T)        

def homogeneous(ttres, recs, vp, vs, solver, damping=0., equality=0., ref={},
    iterate=False):
    """
    Estimate the (x, y) coordinates of the epicenter of an event using
    travel-time residuals between P and S waves and assuming a homogeneous Earth

    Parameters:

    * ttres : array
        Array with the travel-time residuals between S and P waves
    * recs : list of lists
        List with the (x, y) coordinates of the receivers
    * vp : float
        Assumed velocity of P waves
    * vs : float
        Assumed velocity of S waves
    * solver : function
        A non-linear inverse problem solver generated by a factory function
        from a :mod:`~fatiando.inversion` inverse problem solver module.
    * damping : float
        Damping regularizing parameter (i.e., how much damping to apply).
        Must be a positive scalar.
    * equality : float
        Positive scalar regularizing parameter for Equality constraints.
        How much to impose the reference values on the coordinates of the
        epicenter. A high value means strong constraints, a small value means
        loose constraints.
        (see :class:`~fatiando.inversion.regularizer.Equality`).
    * ref : dict
        The reference values for the x or y coordinates (or both). For example,
        to impose that x be close to 8, use ``ref = {'x':8}``. To impose a
        constraint on both x and y, use ``ref = {'x':8, 'y':12}``.
    * iterate : True or False
        If True, will yield the current estimate at each iteration yielded by
        *solver*. In Python terms, ``iterate=True`` transforms this function
        into a generator function.

    Returns:

    * results : list = [estimate, residuals]
        The estimated (x, y) coordinates and the residuals (difference between
        measured and predicted travel-time residuals)
    
    """
    if len(ttres) != len(recs):
        msg = "Must have same number of travel-time residuals and receivers"
        raise ValueError, msg
    if damping < 0:
        raise ValueError, "Damping must be positive"
    if equality < 0:
        raise ValueError, "Equality must be positive"
    for k in ref:
        if k not in ['x', 'y']:
            raise ValueError, "Invalid key in ref: %s" % (str(k))
    dms = [TTRFlat(ttres, recs, vp, vs)]
    regs = []
    if equality:
        reference = {}
        if 'x' in ref:
            reference[0] = ref['x']
        if 'y' in ref:
            reference[1] = ref['y']
        regs.append(inversion.regularizer.Equality(equality, reference))
    if damping:
        regs.append(inversion.regularizer.Damping(damping, 2))
    log.info("Estimating epicenter in 2D assuming a homogeneous Earth:")
    log.info("  equality: %g" % (equality))
    log.info("  reference parameters: %s" % (str(ref)))
    log.info("  damping: %g" % (damping))
    log.info("  iterate: %s" % (str(iterate)))
    if iterate:
        return _iterator(dms, regs, solver)
    else:
        return _solver(dms, regs, solver)
        
def _solver(dms, regs, solver):
    start = time.time()
    try:
        for i, chset in enumerate(solver(dms, regs)):
            continue
    except numpy.linalg.linalg.LinAlgError:
        raise ValueError, ("Oops, the Hessian is a singular matrix." +
                           " Try applying more regularization")
    stop = time.time()
    log.info("  number of iterations: %d" % (i))
    log.info("  final data misfit: %g" % (chset['misfits'][-1]))
    log.info("  final goal function: %g" % (chset['goals'][-1]))
    log.info("  time: %s" % (utils.sec2hms(stop - start)))
    return chset['estimate'], chset['residuals'][0]

def _iterator(dms, regs, solver):
    start = time.time()
    try:
        for i, chset in enumerate(solver(dms, regs)):
            yield chset['estimate'], chset['residuals'][0]
    except numpy.linalg.linalg.LinAlgError:
        raise ValueError, ("Oops, the Hessian is a singular matrix." +
                           " Try applying more regularization")
    stop = time.time()
    log.info("  number of iterations: %d" % (i))
    log.info("  final data misfit: %g" % (chset['misfits'][-1]))
    log.info("  final goal function: %g" % (chset['goals'][-1]))
    log.info("  time: %s" % (utils.sec2hms(stop - start)))

def mapgoal(xs, ys, ttres, recs, vp, vs, damping=0., equality=0., ref={}):
    """
    Make a map of the goal function for a given set of inversion parameters.

    The goal function is define as:

    .. math::

        \\Gamma(\\bar{p}) = \\phi(\\bar{p}) + \\sum\\limits_{r=1}^{R}
        \\mu_r\\theta_r(\\bar{p})

    where :math:`\\phi(\\bar{p})` is the data-misfit function, 
    :math:`\\theta_r(\\bar{p})` is the rth of the R regularizing function used,
    and :math:`\\mu_r` is the rth regularizing parameter.

    Parameters:

    * xs, ys : arrays
        Lists of x and y values where the goal function will be calculated
    * ttres : array
        Array with the travel-time residuals between S and P waves
    * recs : list of lists
        List with the (x, y) coordinates of the receivers
    * vp : float
        Assumed velocity of P waves
    * vs : float
        Assumed velocity of S waves
    * damping : float     
        Damping regularizing parameter (i.e., how much damping to apply).
        Must be a positive scalar.
    * equality : float
        Positive scalar regularizing parameter for Equality constraints.
        How much to impose the reference values on the coordinates of the
        epicenter. A high value means strong constraints, a small value means
        loose constraints.
        (see :class:`~fatiando.inversion.regularizer.Equality`).
    * ref : dict
        The reference values for the x or y coordinates (or both). For example,
        to impose that x be close to 8, use ``ref = {'x':8}``. To impose a
        constraint on both x and y, use ``ref = {'x':8, 'y':12}``.

    Returns:

    * goals : array
        Array with the goal function values
    
    """    
    dm = TTRFlat(ttres, recs, vp, vs)
    reference = {}
    if 'x' in ref:
        reference[0] = ref['x']
    if 'y' in ref:
        reference[1] = ref['y']
    reg1 = inversion.regularizer.Equality(equality, reference)
    reg2 = inversion.regularizer.Damping(damping, 2)
    return numpy.array([dm.get_misfit(ttres - dm.get_predicted(p)) +
                        reg1.value(p) + reg2.value(p) for p in zip(xs, ys)])
