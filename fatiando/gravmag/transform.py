"""
Space domain potential field transformations, like upward continuation,
derivatives and total mass.

**Transformations**

* :func:`~fatiando.gravmag.transform.upcontinue`: Upward continuation of
  potential field data.
* :func:`~fatiando.gravmag.transform.reduce_to_pole`: Reduce the total field
  magnetic anomaly to the pole.
* :func:`~fatiando.gravmag.transform.tga`: Calculate the amplitude of the
  total gradient (also called the analytic signal)

**Derivatives**

* :func:`~fatiando.gravmag.transform.derivx`: Calculate the n-th order
  derivative of a potential field in the x-direction
* :func:`~fatiando.gravmag.transform.derivy`: Calculate the n-th order
  derivative of a potential field in the y-direction
* :func:`~fatiando.gravmag.transform.derivz`: Calculate the n-th order
  derivative of a potential field in the z-direction

----

"""
from __future__ import division
import numpy

from .. import utils


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


def upcontinue(x, y, data, shape, height, method='fft'):
    r"""
    Upward continuation of potential field data.

    Has the option of calculating the continuation through the Fast Fourier
    Transform in the wavenumber domain or through numerical integration of the
    analytical formula below in the space domain:

    .. math::

        g_z(x,y,z) = \\frac{z-z_0}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^
        {\infty} g_z(x',y',z_0) \\frac{1}{[(x-x')^2 + (y-y')^2 + (z-z_0)^2
        ]^{\\frac{3}{2}}} dx' dy'

    For the FFT based continuation. The Fourier transform of the upward
    continued field is calculated using:

    .. math::

        F\{h_{up}\} = F\{h\} e^{-\Delta z |k|}

    and then transformed back to the space domain.

    .. note:: Data needs to be on a regular grid!

    .. note:: Units are SI for all coordinates x, y, z.

    .. note:: be aware of coordinate systems!
        The *x*, *y*, *z* coordinates are:
        x -> North, y -> East and z -> **DOWN**.

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid
    * height : float
        The height increase (delta z) in meters.
    * method : string
        The method used to upward continue. Can be either: ``'fft'`` for FFT
        based continuation or ``'space'`` for the space domain approach.

    Returns:

    * cont : array
        The upward continued data

    """
    assert method in ['fft', 'space'], \
        "Invalid method '{}'".format(method)
    assert x.shape == y.shape, \
        "x and y arrays must have same shape"
    assert height > 0, \
        "Continuation height increase 'height' should be positive"
    if method == 'fft':
        kx, ky = _getfreqs(x, y, data, shape)
        kz = numpy.sqrt(kx**2 + ky**2)
        ft = numpy.fft.fft2(numpy.reshape(data, shape))
        ft_up = numpy.exp(-height*kz)*ft
        cont = numpy.real(numpy.fft.ifft2(ft_up)).ravel()
    elif method == 'space':
        nx, ny = shape
        dx = (x.max() - x.min())/(nx - 1)
        dy = (y.max() - y.min())/(ny - 1)
        area = dx*dy
        deltaz_sqr = (height)**2
        cont = numpy.zeros_like(data)
        for i, j, g in zip(x, y, data):
            cont += g*area*((x - i)**2 + (y - j)**2 + deltaz_sqr)**(-1.5)
        cont *= abs(height)/(2*numpy.pi)
    return cont


def tga(x, y, data, shape):
    """
    Calculate the total gradient amplitude.

    This the same as the `analytic signal`, but we prefer the newer, more
    descriptive nomenclature.

    .. warning::

        If the data is not in SI units, the derivatives will be in
        strange units and so will the total gradient amplitude! I strongly
        recommend converting the data to SI **before** calculating the
        derivative (use one of the unit conversion functions of
        :mod:`fatiando.utils`).

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (ny, nx)
        The shape of the grid

    Returns:

    * tga : 1D-array
        The amplitude of the total gradient

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


def _getfreqs(x, y, data, shape):
    """
    Get two 2D-arrays with the wave numbers in the x and y directions.
    """
    ny, nx = shape
    dx = float(x.max() - x.min()) / float(nx - 1)
    fx = (2*numpy.pi)*numpy.fft.fftfreq(nx, dx)
    dy = float(y.max() - y.min()) / float(ny - 1)
    fy = (2*numpy.pi)*numpy.fft.fftfreq(ny, dy)
    return numpy.meshgrid(fx, fy)


def _deriv(freqs, data, shape, order):
    """
    Calculate a generic derivative using the FFT.
    """
    fgrid = numpy.fft.fft2(numpy.reshape(data, shape))
    deriv = numpy.real(numpy.fft.ifft2((freqs ** order) * fgrid).ravel())
    return deriv
