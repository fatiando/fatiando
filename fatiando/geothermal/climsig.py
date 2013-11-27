r"""
Modeling and inversion of temperature residuals measured in wells due to
temperature perturbations in the surface.

Perturbations can be of two kinds: **abrupt** or **linear**.

Forward modeling of these types of changes is done with functions:

* :func:`~fatiando.geothermal.climsig.abrupt`
* :func:`~fatiando.geothermal.climsig.linear`

Assumeing that the temperature perturbation was abrupt. The residual
temperature at a depth :math:`z_i` in the well at a time :math:`t` after the
perturbation is given by

.. math::

    T_i(z_i) = A \left[1 - \mathrm{erf}\left(
    \frac{z_i}{\sqrt{4\lambda t}}\right)\right]

where :math:`A` is the amplitude of the perturbation, :math:`\lambda` is the
thermal diffusivity of the medium, and :math:`\mathrm{erf}` is the error
function.

For the case of a linear change, the temperature is

.. math::

    T_i(z_i) = A \left[
    \left(1 + 2\frac{z_i^2}{4\lambda t}\right)
    \mathrm{erfc}\left(\frac{z_i}{\sqrt{4\lambda t}}\right) -
    \frac{2}{\sqrt{\pi}}\left(\frac{z_i}{\sqrt{4\lambda t}}\right)
    \mathrm{exp}\left(-\frac{z_i^2}{4\lambda t}\right)
    \right]

Given the temperature measured at different depths, we can **invert** for the
amplitude and age of the change. The available inversion solvers are:

* :class:`~fatiando.geothermal.climsig.SingleChange`: inverts for the
  parameters of a single temperature change. Can use both abrupt and linear
  models. Example:

    >>> import numpy
    >>> zp = numpy.arange(0, 100, 1)
    >>> # For an ABRUPT change
    >>> amp = 2
    >>> age = 100 # Uses years to avoid overflows
    >>> temp = abrupt(amp, age, zp)
    >>> # Run the inversion for the amplitude and time
    >>> # Need an initial estimate because this is a non-linear problem
    >>> initial = [1, 1]
    >>> data = SingleChange(temp, zp, mode='abrupt')
    >>> est_amp, est_age = data.fit(initial)
    >>> print "amp: %.2f  age: %.2f" % (est_amp, est_age)
    amp: 2.00  age: 100.00
    >>> # For a LINEAR change
    >>> amp = 3.45
    >>> age = 52.5
    >>> temp = linear(amp, age, zp)
    >>> data = SingleChange(temp, zp, mode='linear')
    >>> est_amp, est_age = data.fit(initial)
    >>> print "amp: %.2f  age: %.2f" % (est_amp, est_age)
    amp: 3.45  age: 52.50

----

"""
from __future__ import division
import numpy
import scipy.special

from ..inversion.misfit import L2Norm
from ..constants import THERMAL_DIFFUSIVITY_YEAR


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

class SingleChange(L2Norm):
    r"""
    Invert the well temperature data for a single change in temperature.

    The parameters of the change are its amplitude and age.

    See the docstring of :mod:`fatiando.geothermal.climsig` for more
    information and examples.

    Parameters:

    * temp : array
        The temperature profile
    * zp : array
        Depths along the profile
    * mode : string
        The type of change: ``'abrupt'`` for an abrupt change, ``'linear'`` for
        a linear change.
    * diffus : float
        Thermal diffusivity of the medium (in m^2/year)

    Notes:

    For **abrupt** changes, derivatives with respect to the amplitude and age
    are calculated using the formula

    .. math::

        \frac{\partial T_i}{\partial A} = 1 - \mathrm{erf}\left(
        \frac{z_i}{\sqrt{4\lambda t}}\right)

    and

    .. math::

        \frac{\partial T_i}{\partial t} = \frac{A}{t\sqrt{\pi}}
        \left(\frac{z_i}{\sqrt{4\lambda t}}\right)
        \exp\left[-\left(\frac{z_i}{\sqrt{4\lambda t}}\right)^2\right]

    respectively.

    For **linear** changes, derivatives with respect to the age are calculated
    using a 2-point finite difference approximation. Derivatives with respect
    to amplitude are calculate using the formula

    .. math::

        \frac{\partial T_i}{\partial A} =
        \left(1 + 2\frac{z_i^2}{4\lambda t}\right)
        \mathrm{erfc}\left(\frac{z_i}{\sqrt{4\lambda t}}\right) -
        \frac{2}{\sqrt{\pi}}\left(\frac{z_i}{\sqrt{4\lambda t}}\right)
        \mathrm{exp}\left(-\frac{z_i^2}{4\lambda t}\right)

    """

    def __init__(self, temp, zp, mode, diffus=THERMAL_DIFFUSIVITY_YEAR):
        super(SingleChange, self).__init__(temp, nparams=2, islinear=False)
        if len(temp) != len(zp):
            raise ValueError("temp and zp must be of same length")
        if mode not in ['abrupt', 'linear']:
            raise ValueError("Invalid mode: %s. Must be 'abrupt' or 'linear'"
                % (mode))
        self.zp = zp
        self.diffus = float(diffus)
        self.mode = mode

    def _get_predicted(self, p):
        amp, age = p
        if self.mode == 'abrupt':
            return abrupt(amp, age, self.zp, self.diffus)
        if self.mode == 'linear':
            return linear(amp, age, self.zp, self.diffus)

    def _get_jacobian(self, p):
        amp, age = p
        if self.mode == 'abrupt':
            tmp = self.zp/numpy.sqrt(4.*self.diffus*age)
            jac = numpy.transpose([
                abrupt(1., age, self.zp, self.diffus),
                amp*tmp*numpy.exp(-(tmp**2))/(numpy.sqrt(numpy.pi)*age)])
        if self.mode == 'linear':
            delta = 0.5
            at_p = linear(amp, age, self.zp, self.diffus)
            jac = numpy.transpose([
                linear(1., age, self.zp, self.diffus),
                (linear(amp, age + delta, self.zp, self.diffus) -
                 linear(amp, age - delta, self.zp, self.diffus))/(2*delta)])
        return jac
