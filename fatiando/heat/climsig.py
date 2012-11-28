"""
Modeling and inversion of temperature residuals measured in wells due to
temperature perturbations in the surface.

Perturbations can be of two kinds:

**Abrupt**

* :func:`~fatiando.heat.climsig.abrupt`
* :func:`~fatiando.heat.climsig.iabrupt`

Assumes that the temperature perturbation was abrupt. The residual temperature
at a depth :math:`z_i` in the well at a time :math:`t` after the perturbation is
given by

.. math::

    T_i(z_i) = A \\left[1 - \\mathrm{erf}\\left(
    \\frac{z_i}{\\sqrt{4\\lambda t}}\\right)\\right]

where :math:`A` is the amplitude of the perturbation, :math:`\\lambda` is the
thermal diffusivity of the medium, and :math:`\\mathrm{erf}` is the error
function.

Example of inverting for the amplitude and time since the perturbation using
synthetic data::

    >>> import numpy
    >>> import fatiando as ft
    >>> # Generate the sythetic data along a well
    >>> zp = numpy.arange(0, 100, 1)
    >>> amp = 2
    >>> age = 100 # Uses years to avoid overflows
    >>> temp = ft.heat.climsig.abrupt(amp, age, zp)
    >>> # Run the inversion for the amplitude and time
    >>> p, residuals = ft.heat.climsig.iabrupt(temp, zp)
    >>> print "amp: %.2f  age: %.2f" % (p[0], p[1])
    amp: 2.00  age: 100.00

**Linear**

* :func:`~fatiando.heat.climsig.linear`
* :func:`~fatiando.heat.climsig.ilinear`

Assumes that the temperature perturbation was linear with time. The residual
temperature at a depth :math:`z_i` in the well at a time :math:`t` after the
perturbation was started is given by

.. math::

    T_i(z_i) = A \\left[
    \\left(1 + 2\\frac{z_i^2}{4\\lambda t}\\right)
    \\mathrm{erfc}\\left(\\frac{z_i}{\\sqrt{4\\lambda t}}\\right) -
    \\frac{2}{\\sqrt{\\pi}}\\left(\\frac{z_i}{\\sqrt{4\\lambda t}}\\right)
    \\mathrm{exp}\\left(-\\frac{z_i^2}{4\\lambda t}\\right)
    \\right]

where :math:`A` is the amplitude of the perturbation, :math:`\\lambda` is the
thermal diffusivity of the medium, and :math:`\\mathrm{erf}` is the error
function.

Example of inverting for the amplitude and time since the perturbation using
synthetic data::

    >>> import numpy
    >>> import fatiando as ft
    >>> # Generate the sythetic data along a well
    >>> zp = numpy.arange(0, 100, 1)
    >>> amp = 3.45
    >>> age = 52.5 # Uses years to avoid overflows
    >>> temp = ft.heat.climsig.linear(amp, age, zp)
    >>> # Run the inversion for the amplitude and time
    >>> p, residuals = ft.heat.climsig.ilinear(temp, zp)
    >>> print "amp: %.2f  age: %.2f" % (p[0], p[1])
    amp: 3.45  age: 52.50

----

"""

import time
import itertools
import numpy
import scipy.special

import fatiando.logger
from fatiando import inversion, utils
from fatiando.constants import THERMAL_DIFFUSIVITY_YEAR

log = fatiando.logger.dummy('fatiando.heat.climsig')

class AbruptDM(inversion.datamodule.DataModule):
    """
    Data module for a single abrupt temperature perturbation.

    Packs the necessary data for the inversion.

    Derivatives with respect to the amplitude and age are calculated using the
    formula

    .. math::

        \\frac{\\partial T_i}{\\partial A} = 1 - \\mathrm{erf}\\left(
        \\frac{z_i}{\\sqrt{4\\lambda t}}\\right)

    and

    .. math::

        \\frac{\\partial T_i}{\\partial t} = \\frac{A}{t\\sqrt{\\pi}}
        \\left(\\frac{z_i}{\\sqrt{4\\lambda t}}\\right)
        \\exp\\left[-\\left(\\frac{z_i}{\\sqrt{4\\lambda t}}\\right)^2\\right]

    The Hessian matrix is calculated using a Gauss-Newton approximation.

    Parameters:

    * temp : array
        The temperature profile
    * zp : array
        Depths along the profile
    * diffus : float
        Thermal diffusivity of the medium (in m^2/year)

    """

    def __init__(self, temp, zp, diffus=THERMAL_DIFFUSIVITY_YEAR):
        if len(temp) != len(zp):
            raise ValueError, "temp and zp must be of same length"
        inversion.datamodule.DataModule.__init__(self, temp)
        self.temp = numpy.array(temp, dtype='f')
        self.zp = numpy.array(zp, dtype='f')
        self.diffus = float(diffus)

    def get_predicted(self, p):
        amp, age = p
        return abrupt(amp, age, self.zp, self.diffus)

    def sum_gradient(self, gradient, p, residuals):
        amp, age = p
        tmp = self.zp/numpy.sqrt(4.*self.diffus*age)
        jact = amp*tmp*numpy.exp(-(tmp**2))/(numpy.sqrt(numpy.pi)*age)
        jacA = 1. - scipy.special.erf(tmp)
        self.jac_T = numpy.array([jacA, jact])
        return gradient - 2.*numpy.dot(self.jac_T, residuals)

    def sum_hessian(self, hessian, p):
        return hessian + 2*numpy.dot(self.jac_T, self.jac_T.T)

def abrupt(amp, age, zp, diffus=THERMAL_DIFFUSIVITY_YEAR):
    """
    Calculate the residual temperature profile in depth due to an abrupt
    temperature perturbation.

    Parameters:

    * amp : float
        Amplitude of the perturbation (in C)
    * age : float
        Time since the perturbation occured (in years)
    * zp : array
        Arry with the depths of computation points along the well (in meters)
    * diffus : float
        Thermal diffusivity of the medium (in m^2/year)

    See the default values for the thermal diffusivity in
    :mod:`fatiando.constants`.

    Returns

    * temp : array
        The residual temperatures measured along the well

    """
    return amp*(1. - scipy.special.erf(zp/numpy.sqrt(4.*diffus*age)))

def iabrupt(temp, zp, solver=None, diffus=THERMAL_DIFFUSIVITY_YEAR, iterate=False):
    """
    Invert the residual temperature profile to estimate the amplitude and age
    of an abrupt temperature perturbation.

    Parameters:

    * temp : array
        The temperature profile
    * zp : array
        The depths along the profile
    * solver : function or None
        A non-linear inverse problem solver generated by a factory function
        from a :mod:`fatiando.inversion` inverse problem solver module. If
        None, will use the default solver.
    * diffus : float
        Thermal diffusivity of the medium (in m^2/year)
    * iterate : True or False
        If True, will yield the current estimate at each iteration yielded by
        *solver*. In Python terms, ``iterate=True`` transforms this function
        into a generator function.

    See the default values for the thermal diffusivity in
    :mod:`fatiando.constants`.

    Returns:

    * results : list = [p, residuals]
        The estimated paramter vector ``p = [amp, age]`` and the residuals (fit)
        produced by the inversion. The residuals are the observed data minus the
        data predicted by the estimated parameters.

    """
    log.info("Estimating amplitude and age of an abrupt perturbation:")
    log.info("  thermal diffusivity: %g" % (diffus))
    log.info("  number of data: %d" % (len(temp)))
    log.info("  iterate: %s" % (str(iterate)))
    dms = [AbruptDM(temp, zp, diffus)]
    if solver is None:
        solver = inversion.gradient.levmarq(initial=[1,1])
    if iterate:
        return _iterator(dms, solver, log)
    else:
        return _solver(dms, solver, log)

class LinearDM(inversion.datamodule.DataModule):
    """
    Data module for a single linear temperature perturbation.

    Packs the necessary data for the inversion.

    Derivatives with respect to the age are calculated using a 2-point finite
    difference approximation. Derivatives with respect to amplitude are
    calculate using the formula

    .. math::

        \\frac{\\partial T_i}{\\partial A} =
        \\left(1 + 2\\frac{z_i^2}{4\\lambda t}\\right)
        \\mathrm{erfc}\\left(\\frac{z_i}{\\sqrt{4\\lambda t}}\\right) -
        \\frac{2}{\\sqrt{\\pi}}\\left(\\frac{z_i}{\\sqrt{4\\lambda t}}\\right)
        \\mathrm{exp}\\left(-\\frac{z_i^2}{4\\lambda t}\\right)

    The Hessian matrix is calculated using a Gauss-Newton approximation.

    Parameters:

    * temp : array
        The temperature profile
    * zp : array
        The depths along the profile
    * diffus : float
        Thermal diffusivity of the medium (in m^2/year)

    """

    def __init__(self, temp, zp, diffus=THERMAL_DIFFUSIVITY_YEAR):
        if len(temp) != len(zp):
            raise ValueError, "temp and zp must be of same length"
        inversion.datamodule.DataModule.__init__(self, temp)
        self.temp = numpy.array(temp, dtype='f')
        self.zp = numpy.array(zp, dtype='f')
        self.diffus = float(diffus)

    def get_predicted(self, p):
        amp, age = p
        return linear(amp, age, self.zp, self.diffus)

    def sum_gradient(self, gradient, p, residuals):
        amp, age = p
        delta = 0.1
        at_p = linear(amp, age, self.zp, self.diffus)
        jact = (linear(amp, age + delta, self.zp, self.diffus) - at_p)/delta
        jacA = linear(1., age, self.zp, self.diffus)
        self.jac_T = numpy.array([jacA, jact])
        return gradient - 2.*numpy.dot(self.jac_T, residuals)

    def sum_hessian(self, hessian, p):
        return hessian + 2*numpy.dot(self.jac_T, self.jac_T.T)

def linear(amp, age, zp, diffus=THERMAL_DIFFUSIVITY_YEAR):
    """
    Calculate the residual temperature profile in depth due to a linear
    temperature perturbation.

    Parameters:

    * amp : float
        Amplitude of the perturbation (in C)
    * age : float
        Time since the perturbation occured (in years)
    * zp : array
        The depths of computation points along the well (in meters)
    * diffus : float
        Thermal diffusivity of the medium (in m^2/year)

    See the default values for the thermal diffusivity in
    :mod:`fatiando.constants`.

    Returns

    * temp : array
        The residual temperatures measured along the well

    """
    tmp = zp/numpy.sqrt(4.*diffus*age)
    res = amp*((1. + 2*tmp**2)*scipy.special.erfc(tmp)
               - 2./numpy.sqrt(numpy.pi)*tmp*numpy.exp(-tmp**2))
    return res

def ilinear(temp, zp, solver=None, diffus=THERMAL_DIFFUSIVITY_YEAR, iterate=False):
    """
    Invert the residual temperature profile to estimate the amplitude and age
    of a linear temperature perturbation.

    Parameters:

    * temp : array
        Array with the temperature profile
    * zp : array
        Array with the depths along the profile
    * solver : function
        A non-linear inverse problem solver generated by a factory function
        from a :mod:`~fatiando.inversion` inverse problem solver module.If
        None, will use the default solver.
    * diffus : float
        Thermal diffusivity of the medium (in m^2/year)
    * iterate : True or False
        If True, will yield the current estimate at each iteration yielded by
        *solver*. In Python terms, ``iterate=True`` transforms this function
        into a generator function.

    See the default values for the thermal diffusivity in
    :mod:`fatiando.constants`.

    Returns:

    * results : list = [p, residuals]
        The estimated paramter vector ``p = [amp, age]`` and the residuals (fit)
        produced by the inversion. The residuals are the observed data minus the
        data predicted by the estimated parameters.

    """
    log.info("Estimating amplitude and age of a linear perturbation:")
    log.info("  thermal diffusivity: %g" % (diffus))
    log.info("  number of data: %d" % (len(temp)))
    log.info("  iterate: %s" % (str(iterate)))
    dms = [LinearDM(temp, zp, diffus)]
    if solver is None:
        solver = inversion.gradient.levmarq(initial=[1,1])
    if iterate:
        return _iterator(dms, solver, log)
    else:
        return _solver(dms, solver, log)

def _solver(dms, solver, log):
    start = time.time()
    try:
        for i, chset in enumerate(solver(dms, [])):
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

def _iterator(dms, solver, log):
    start = time.time()
    try:
        for i, chset in enumerate(solver(dms, [])):
            yield chset['estimate'], chset['residuals'][0]
    except numpy.linalg.linalg.LinAlgError:
        raise ValueError, ("Oops, the Hessian is a singular matrix." +
                           " Try applying more regularization")
    stop = time.time()
    log.info("  number of iterations: %d" % (i))
    log.info("  final data misfit: %g" % (chset['misfits'][-1]))
    log.info("  final goal function: %g" % (chset['goals'][-1]))
    log.info("  time: %s" % (utils.sec2hms(stop - start)))
