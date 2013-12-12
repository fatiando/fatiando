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
  models.


----

"""
from __future__ import division
import numpy
import scipy.special

from ..inversion.base import Misfit
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

class SingleChange(Misfit):
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

    Example with synthetic data:

        >>> import numpy
        >>> zp = numpy.arange(0, 100, 1)
        >>> # For an ABRUPT change
        >>> amp = 2
        >>> age = 100 # Uses years to avoid overflows
        >>> temp = abrupt(amp, age, zp)
        >>> # Run the inversion for the amplitude and time
        >>> solver = SingleChange(temp, zp, mode='abrupt')
        >>> solver
        SingleChange(
            temp=array([ 2.   ,  1.98 ,  1.96 ,  1.94 ,  1.92 ,  1.9  ,
                1.88 ,  1.86 ,  1.84 ,  1.82 ,  1.8  ,  1.78 ,
                1.76 ,  1.74 ,  1.72 ,  1.7  ,  1.681,  1.661,
                1.642,  1.622,  1.602,  1.583,  1.564,  1.544,
                1.525,  1.506,  1.487,  1.468,  1.449,  1.43 ,
                1.411,  1.393,  1.374,  1.356,  1.337,  1.319,
                1.301,  1.283,  1.265,  1.247,  1.229,  1.212,
                1.194,  1.177,  1.159,  1.142,  1.125,  1.108,
                1.091,  1.075,  1.058,  1.042,  1.026,  1.009,
                0.993,  0.977,  0.962,  0.946,  0.931,  0.915,
                0.9  ,  0.885,  0.87 ,  0.856,  0.841,  0.827,
                0.812,  0.798,  0.784,  0.77 ,  0.757,  0.743,
                0.73 ,  0.716,  0.703,  0.69 ,  0.678,  0.665,
                0.652,  0.64 ,  0.628,  0.616,  0.604,  0.592,
                0.581,  0.569,  0.558,  0.547,  0.536,  0.525,
                0.515,  0.504,  0.494,  0.484,  0.473,  0.464,
                0.454,  0.444,  0.435,  0.425]),
            zp=array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
               26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
               39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
               52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
               65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
               78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
               91, 92, 93, 94, 95, 96, 97, 98, 99]),
            mode='abrupt',
            diffus=31.5576)
        >>> # Need an initial estimate because this is a non-linear problem
        >>> initial = [1, 1]
        >>> est_amp, est_age = solver.fit(initial=initial)
        >>> print "amp: %.2f  age: %.2f" % (est_amp, est_age)
        amp: 2.00  age: 100.00
        >>> # For a LINEAR change
        >>> amp = 3.45
        >>> age = 52.5
        >>> temp = linear(amp, age, zp)
        >>> solver = SingleChange(temp, zp, mode='linear')
        >>> est_amp, est_age = solver.fit(initial=initial)
        >>> print "amp: %.2f  age: %.2f" % (est_amp, est_age)
        amp: 3.45  age: 52.50

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
        if len(temp) != len(zp):
            raise ValueError("temp and zp must be of same length")
        if mode not in ['abrupt', 'linear']:
            raise ValueError("Invalid mode: %s. Must be 'abrupt' or 'linear'"
                % (mode))
        super(SingleChange, self).__init__(
            data=temp,
            positional=dict(zp=zp),
            model=dict(diffus=float(diffus), mode=mode),
            nparams=2, islinear=False)

    def __repr__(self):
        lw = 60
        prec = 3
        text = '\n'.join([
            'SingleChange(',
            '    temp=%s,' % (numpy.array_repr(
                self.data, max_line_width=lw, precision=prec)),
            '    zp=%s,' % (numpy.array_repr(
                self.positional['zp'], max_line_width=lw, precision=prec)),
            "    mode='%s'," % (self.model['mode']),
            '    diffus=%g)' % (self.model['diffus'])])
        return text

    def _get_predicted(self, p):
        amp, age = p
        zp = self.positional['zp']
        diffus = self.model['diffus']
        if self.model['mode'] == 'abrupt':
            return abrupt(amp, age, zp, diffus)
        if self.model['mode'] == 'linear':
            return linear(amp, age, zp, diffus)

    def _get_jacobian(self, p):
        amp, age = p
        zp = self.positional['zp']
        diffus = self.model['diffus']
        mode = self.model['mode']
        if mode == 'abrupt':
            tmp = zp/numpy.sqrt(4.*diffus*age)
            jac = numpy.transpose([
                abrupt(1., age, zp, diffus),
                amp*tmp*numpy.exp(-(tmp**2))/(numpy.sqrt(numpy.pi)*age)])
        if mode == 'linear':
            delta = 0.5
            at_p = linear(amp, age, zp, diffus)
            jac = numpy.transpose([
                linear(1., age, zp, diffus),
                (linear(amp, age + delta, zp, diffus) -
                 linear(amp, age - delta, zp, diffus))/(2*delta)])
        return jac
