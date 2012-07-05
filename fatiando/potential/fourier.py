"""
Potential field processing using the Fast Fourier Transform.

Generally requires gridded data to work.

**Transformations**

* :func:`~fatiando.potential.fourier.deriv`: Calculate the n-th order derivative
  of a 2D function in a given direction

----
"""

import numpy


def derivx(x, y, data, shape, order=1):
    """
    Calculate the derivative of a 2D function in the x direction.
    """
    Fx = _getfreqs(x, y, data, shape)[0].astype('complex')
    # Multiply by 1j because I don't multiply it in _deriv (this way _deriv can
    # be used for the z derivative as well)
    return _deriv(Fx*1j, data, shape, order)

def derivy(x, y, data, shape, order=1):
    """
    Calculate the derivative of a 2D function in the x direction.
    """
    Fy= _getfreqs(x, y, data, shape)[1].astype('complex')
    # Multiply by 1j because I don't multiply it in _deriv (this way _deriv can
    # be used for the z derivative as well)
    return _deriv(Fy*1j, data, shape, order)

def derivz(x, y, data, shape, order=1):
    """
    Calculate the derivative of a 2D function in the x direction.
    """
    Fx, Fy = _getfreqs(x, y, data, shape)
    freqs = numpy.sqrt(Fx**2 + Fy**2)
    return _deriv(freqs, data, shape, order)

def _getfreqs(x, y, data, shape):
    ny, nx = shape
    dx = float(x.max() - x.min())/float(nx - 1)
    fx = numpy.fft.fftfreq(nx, dx)
    dy = float(y.max() - y.min())/float(ny - 1)
    fy = numpy.fft.fftfreq(ny, dy)
    return numpy.meshgrid(fx, fy)

def _deriv(freqs, data, shape, order):
    fgrid = (2.*numpy.pi)*numpy.fft.fft2(numpy.reshape(data, shape))
    deriv = numpy.real(numpy.fft.ifft2((freqs**order)*fgrid).ravel())
    return deriv
