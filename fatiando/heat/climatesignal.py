# Copyright 2012 The Fatiando a Terra Development Team
#
# This file is part of Fatiando a Terra.
#
# Fatiando a Terra is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Fatiando a Terra is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Fatiando a Terra.  If not, see <http://www.gnu.org/licenses/>.
"""
Modeling and inversion of temperature residuals measured in wells due to
temperature perturbations in the surface.

Perturbations can be of two kinds:

**ABRUPT**

* :func:`fatiando.heat.climatesignal.abrupt`
* :func:`fatiando.heat.climatesignal.invert_abrupt`

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
    >>> from fatiando.inversion.gradient import levmarq
    >>> from fatiando import utils
    >>> # Generate the sythetic data along a well 
    >>> zp = numpy.arange(0, 100, 1)
    >>> amp = 2
    >>> age = 100 # Uses years to avoid overflows
    >>> temp = abrupt(amp, age, zp) # Use the default diffusivity
    >>> # Run the inversion for the amplitude and time
    >>> solver = levmarq(initial=(10, 50))
    >>> p, residuals = invert_abrupt(temp, zp, solver)
    >>> print "amp: %.2f  age: %.2f" % (p[0], p[1])
    amp: 2.00  age: 100.00

**LINEAR**

* :func:`fatiando.heat.climatesignal.linear`
* :func:`fatiando.heat.climatesignal.invert_linear`

----

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 30-Jan-2012'


import time
import itertools
import numpy

from fatiando.heat import _climatesignal
from fatiando import inversion, utils, logger

log = logger.dummy()

 
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
    
    * temp
        Array with the temperature profile
    * zp
        Array with the depths along the profile
    * diffus
        Thermal diffusivity of the medium (in m^2/year)

    """

    # The default diffusivity is 0.000001 m^2/s = 31.5576 m^2/year
    def __init__(self, temp, zp, diffus=31.5576):
        if len(temp) != len(zp):
            raise ValueError, "temp and zp must be of same length"
        inversion.datamodule.DataModule.__init__(self, temp)
        log.info("  thermal diffusivity: %g" % (diffus))
        log.info("  number of data: %d" % (len(temp)))
        self.temp = numpy.array(temp, dtype='f')
        self.zp = numpy.array(zp, dtype='f')
        self.diffus = float(diffus)        

    def get_predicted(self, p):
        amp, age = p
        return _climatesignal.climatesignal_abrupt(self.diffus, amp, age,
            self.zp)

    def sum_gradient(self, gradient, p, residuals):
        amp, age = p
        tmp = self.zp/numpy.sqrt(4.*self.diffus*age)        
        jact = amp*tmp*numpy.exp(-(tmp**2))/(numpy.sqrt(numpy.pi)*age)
        jacA = _climatesignal.climatesignal_abrupt(self.diffus, 1., age,
                                                   self.zp)
        self.jac_T = numpy.array([jacA, jact])
        return gradient - 2.*numpy.dot(self.jac_T, residuals)

    def sum_hessian(self, hessian, p):
        return hessian + 2*numpy.dot(self.jac_T, self.jac_T.T)
    
def abrupt(amp, age, zp, diffus=31.5576):
    """
    Calculate the residual temperature profile in depth due to an abrupt
    temperature perturbation.

    Parameters:

    * amp
        Amplitude of the perturbation (in C)
    * age
        Time since the perturbation occured (in seconds)
    * zp
        Arry with the depths of computation points along the well (in meters)
    * diffus
        Thermal diffusivity of the medium (in m^2/year)

    The default diffusivity is 0.000001 m^2/s = 31.5576 m^2/year

    Returns

    * temp
        Array with the residual temperatures measured along the well
        
    """
    return _climatesignal.climatesignal_abrupt(float(diffus), float(amp),
        float(age), numpy.array(zp, dtype='f'))
    
def invert_abrupt(temp, zp, solver, diffus=31.5576, iterate=False):
    """
    Invert the residual temperature profile to estimate the amplitude and age
    of an abrupt temperature perturbation.

    Parameters:

    * temp
        Array with the temperature profile
    * zp
        Array with the depths along the profile
    * solver
        A non-linear inverse problem solver generated by a factory function
        from a :mod:`fatiando.inversion` inverse problem solver module.
    * diffus
        Thermal diffusivity of the medium (in m^2/year)
    * iterate
        If True, will yield the current estimate at each iteration yielded by
        *solver*. In Python terms, ``iterate=True`` transforms this function
        into a generator function.

    The default diffusivity is 0.000001 m^2/s = 31.5576 m^2/year

    Returns:

    * [p, residuals]
        The estimated paramter vector ``p = [amp, age]`` and the residuals (fit)
        produced by the inversion. The residuals are the observed data minus the
        data predicted by the estimated parameters.

    """
    log.info("Estimating amplitude and age of an abrupt perturbation:")
    log.info("  iterate: %s" % (str(iterate)))
    dms = [AbruptDM(temp, zp, diffus)]
    if iterate:
        return _iterator(dms, solver)
    else:
        return _solver(dms, solver)

def _solver(dms, solver):
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

def _iterator(dms, solver):
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
    
def _test():
    import doctest
    doctest.testmod()
    print "doctest finished"

if __name__ == '__main__':
    _test()
