"""
Potential field processing using the Fast Fourier Transform

.. note:: Requires gridded data to work!

**Derivatives**

* :func:`~fatiando.gravmag.fourier.derivx`: Calculate the n-th order
  derivative of a potential field in the x-direction
* :func:`~fatiando.gravmag.fourier.derivy`: Calculate the n-th order
  derivative of a potential field in the y-direction
* :func:`~fatiando.gravmag.fourier.derivz`: Calculate the n-th order
  derivative of a potential field in the z-direction

.. warning::

    Calculating the x-derivative of the x-component of the gravitational
    attraction (gx) fails for some reason.


**Transformations**

* :func:`~fatiando.gravmag.fourier.ansig`: Calculate the amplitude of the
  analytic signal
* :func:`~fatiando.gravmag.fourier.upcontinue`: Upward continuation of
  potential field data.
* :func:`~fatiando.gravmag.fourier.reduce_to_pole`: Reduce the total field
  magnetic anomaly to the pole.


----
"""
from __future__ import division
import numpy

from .. import utils


def ansig(x, y, data, shape):
    """
    Calculate the amplitude of the analytic signal of the data.

    .. warning::

        If the data is not in SI units, the derivatives will be in
        strange units and so will the analytic signal! I strongly recommend
        converting the data to SI **before** calculating the derivative (use
        one of the unit conversion functions of :mod:`fatiando.utils`).

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (ny, nx)
        The shape of the grid

    Returns:

    * ansig : 1D-array
        The amplitude of the analytic signal

    """
    dx = derivx(x, y, data, shape)
    dy = derivy(x, y, data, shape)
    dz = derivz(x, y, data, shape)
    res = numpy.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return res


def derivx(x, y, data, shape, order=1):
    """
    Calculate the derivative of a potential field in the x direction.

    .. warning::

        If the data is not in SI units, the derivative will be in
        strange units! I strongly recommend converting the data to SI
        **before** calculating the derivative (use one of the unit conversion
        functions of :mod:`fatiando.utils`). This way the derivative will be in
        SI units and can be easily converted to what unit you want.

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (ny, nx)
        The shape of the grid
    * order : int
        The order of the derivative

    Returns:

    * deriv : 1D-array
        The derivative

    """
    Fx = _getfreqs(x, y, data, shape)[0].astype('complex')
    # Multiply by 1j because I don't multiply it in _deriv (this way _deriv can
    # be used for the z derivative as well)
    return _deriv(Fx*1j, data, shape, order)


def derivy(x, y, data, shape, order=1):
    """
    Calculate the derivative of a potential field in the y direction.

    .. warning::

        If the data is not in SI units, the derivative will be in
        strange units! I strongly recommend converting the data to SI
        **before** calculating the derivative (use one of the unit conversion
        functions of :mod:`fatiando.utils`). This way the derivative will be in
        SI units and can be easily converted to what unit you want.

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (ny, nx)
        The shape of the grid
    * order : int
        The order of the derivative

    Returns:

    * deriv : 1D-array
        The derivative

    """
    Fy = _getfreqs(x, y, data, shape)[1].astype('complex')
    # Multiply by 1j because I don't multiply it in _deriv (this way _deriv can
    # be used for the z derivative as well)
    return _deriv(Fy*1j, data, shape, order)


def derivz(x, y, data, shape, order=1):
    """
    Calculate the derivative of a potential field in the z direction.

    .. warning::

        If the data is not in SI units, the derivative will be in
        strange units! I strongly recommend converting the data to SI
        **before** calculating the derivative (use one of the unit conversion
        functions of :mod:`fatiando.utils`). This way the derivative will be in
        SI units and can be easily converted to what unit you want.

    .. warning::

        There seems to be a (small) systematic error in the z-derivative. The
        formula is correct and this might be a numerial error from computing
        with the FFT. See issue
        `#167 <https://github.com/fatiando/fatiando/issues/167>`__.


    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (ny, nx)
        The shape of the grid
    * order : int
        The order of the derivative

    Returns:

    * deriv : 1D-array
        The derivative

    """
    Fx, Fy = _getfreqs(x, y, data, shape)
    freqs = numpy.sqrt(Fx**2 + Fy**2)
    return _deriv(freqs, data, shape, order)


def _getfreqs(x, y, data, shape):
    """
    Get two 2D-arrays with the wave numbers in the x and y directions. Wave
    numbers are like angular frequencies (with the 2*Pi factor).
    """
    ny, nx = shape
    dx = float(x.max() - x.min()) / float(nx - 1)
    fx = 2*numpy.pi*numpy.fft.fftfreq(nx, dx)
    dy = float(y.max() - y.min()) / float(ny - 1)
    fy = 2*numpy.pi*numpy.fft.fftfreq(ny, dy)
    return numpy.meshgrid(fx, fy)


def _deriv(freqs, data, shape, order):
    """
    Calculate a generic derivative using the FFT.

    *freqs* are the frequencies (not angular frequencies). In the case of x-,
    and y-derivatives, the should be multiplied by 1j (the imaginary number).
    """
    ft = numpy.fft.fft2(numpy.reshape(data, shape))
    ft_deriv = (freqs**order)*ft
    deriv = numpy.real(numpy.fft.ifft2(ft_deriv)).ravel()
    return deriv


def upcontinue(x, y, data, shape, height):
    r"""
    Upward continuation of potential field data.

    The Fourier transform of the upward continued field is:

    .. math::

        F\{h_{up}\} = F\{h\} e^{-\Delta z |k|}

    """
    kx, ky = _getfreqs(x, y, data, shape)
    kz = numpy.sqrt(kx**2 + ky**2)
    ft = numpy.fft.fft2(numpy.reshape(data, shape))
    ft_up = numpy.exp(-height*kz)*ft
    data_up = numpy.real(numpy.fft.ifft2(ft_up)).ravel()
    return data_up


def reduce_to_pole(x, y, data, shape, inc, dec, sinc=None, sdec=None):
    """
    Parameters:

    * x, y : 1d-arrays
        The x, y, z coordinates of each data point.
    * data : 1d-array
        The total field anomaly data at each point.
    * inc, dec : floats
        The inclination and declination of the inducing field
    * sinc, sdec : None or floats
        The inclination and declination of the equivalent layer. Use these if
        there is remanent magnetization and the total magnetization of the
        layer if different from the induced magnetization.
        If there is only induced magnetization, use None
    """
    fx, fy, fz = utils.ang2vec(1, inc, dec)
    if sinc is None or sdec is None:
        mx, my, mz = fx, fy, fz
    else:
        mx, my, mz = utils.ang2vec(1, sinc, sdec)
    kx, ky = [k for k in _getfreqs(x, y, data, shape)]
    kz_sqr = kx**2 + ky**2
    a1 = mz*fz - mx*fx
    a2 = mz*fz - my*fy
    a3 = -my*fx - mx*fy
    b1 = mx*fz + mz*fx
    b2 = my*fz + mz*fy
    # The division gives a RuntimeWarning because of the zero frequency term.
    # This suppresses the warning.
    with numpy.errstate(divide='ignore', invalid='ignore'):
        rtp = (kz_sqr)/(a1*kx**2 + a2*ky**2 + a3*kx*ky +
                        1j*numpy.sqrt(kz_sqr)*(b1*kx + b2*ky))
    rtp[0, 0] = 0
    ft = numpy.fft.fft2(numpy.reshape(data, shape))
    ft_pole = ft*rtp
    data_pole = numpy.real(numpy.fft.ifft2(ft_pole)).ravel()
    return data_pole
