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

**Radial Average**

* :func:`~fatiando.gravmag.fourier.radial_average`:Calculate the Radial
  Average of a Power Spectrum or any function of it (i.e. Log Power
  Spectrum).

**Transformations**

* :func:`~fatiando.gravmag.fourier.ansig`: Calculate the amplitude of the
  analytic signal

----
"""

import numpy


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
    return _deriv(Fx * 1j, data, shape, order)


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
    return _deriv(Fy * 1j, data, shape, order)


def derivz(x, y, data, shape, order=1):
    """
    Calculate the derivative of a potential field in the z direction.

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
    Fx, Fy = _getfreqs(x, y, data, shape)
    freqs = numpy.sqrt(Fx ** 2 + Fy ** 2)
    return _deriv(freqs, data, shape, order)


def radial_average(spectrum, dx=1.0):
    """
    Performs a radial average of the spectrum grid using concentric
    rings, and calculates the standard deviation for each average.

    Parameters:

    * spectrum: real 2D-array
        Power Spectrum on regular grid or any function of it
        (i.e. Log Power Spectrum).
        The grid must be square, equally spaced and must have
        odd number of points per axis.

    * dx: float
        Sampling distance (default = 1)

    .. warning ::

        The spectrum has to be in the form as the output of fft2,
        DON'T USE np.fft.fftshift.

    Returns:

    * freqs: 1D-array
        Array containing the frequency values for each averaged ring.
    * averages: 1D-array
        Radial averages of the spectrum.
    * rms: 1D-array
        Error bar for each average.
    """

    ny, nx = spectrum.shape
    assert ny == nx, "The spectrum grid isn't square."
    assert nx % 2 != 0, \
        "The spectrum grid must have odd number of points per axis."

    freqs = np.fft.fftfreq(nx, dx)
    fx, fy = np.meshgrid(freqs, freqs)
    f = np.sqrt(fx**2 + fy**2)
    df = 1/float(dx*nx)

    averages = [spectrum[f == 0][0]]
    rms = [0]
    for freq in freqs[1:nx/2 + 1]:
        ring_points = spectrum[(f <= freq + df/2.) & (f > freq - df/2.)]
        ring_n = float(len(ring_points))
        ring_average = np.sum(ring_points)/ring_n
        ring_rms = np.sqrt(np.sum((ring_points - ring_average)**2)/ring_n**2)
        averages.append(ring_average)
        rms.append(ring_rms)

    return freqs[:nx/2+1], np.array(averages), np.array(rms)


def _getfreqs(x, y, data, shape):
    """
    Get two 2D-arrays with the wave numbers in the x and y directions.
    """
    ny, nx = shape
    dx = float(x.max() - x.min()) / float(nx - 1)
    fx = numpy.fft.fftfreq(nx, dx)
    dy = float(y.max() - y.min()) / float(ny - 1)
    fy = numpy.fft.fftfreq(ny, dy)
    return numpy.meshgrid(fx, fy)


def _deriv(freqs, data, shape, order):
    """
    Calculate a generic derivative using the FFT.
    """
    fgrid = (2. * numpy.pi) * numpy.fft.fft2(numpy.reshape(data, shape))
    deriv = numpy.real(numpy.fft.ifft2((freqs ** order) * fgrid).ravel())
    return deriv
