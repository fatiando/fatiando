"""
Potential field processing using the Fast Fourier Transform

.. note:: Requires gridded data to work!

**Derivatives**

* :func:`~fatiando.potential.fourier.derivx`: Calculate the n-th order
  derivative of a potential field in the x-direction
* :func:`~fatiando.potential.fourier.derivy`: Calculate the n-th order
  derivative of a potential field in the y-direction
* :func:`~fatiando.potential.fourier.derivz`: Calculate the n-th order
  derivative of a potential field in the z-direction

----
"""

import numpy


def derivx(x, y, data, shape, order=1):
    """
    Calculate the derivative of a potential field in the x direction.

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
    Fy= _getfreqs(x, y, data, shape)[1].astype('complex')
    # Multiply by 1j because I don't multiply it in _deriv (this way _deriv can
    # be used for the z derivative as well)
    return _deriv(Fy*1j, data, shape, order)

def derivz(x, y, data, shape, order=1):
    """
    Calculate the derivative of a potential field in the z direction.

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
    Get two 2D-arrays with the wave numbers in the x and y directions.
    """
    ny, nx = shape
    dx = float(x.max() - x.min())/float(nx - 1)
    fx = numpy.fft.fftfreq(nx, dx)
    dy = float(y.max() - y.min())/float(ny - 1)
    fy = numpy.fft.fftfreq(ny, dy)
    return numpy.meshgrid(fx, fy)

def _deriv(freqs, data, shape, order):
    """
    Calculate a generic derivative using the FFT.
    """
    fgrid = (2.*numpy.pi)*numpy.fft.fft2(numpy.reshape(data, shape))
    deriv = numpy.real(numpy.fft.ifft2((freqs**order)*fgrid).ravel())
    return deriv
