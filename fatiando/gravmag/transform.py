"""
Potential field transformations, like upward continuation and derivatives.

.. note:: Most, if not all, functions here required gridded data.

**Transformations**

* :func:`~fatiando.gravmag.transform.upcontinue`: Upward continuation of
  gridded potential field data on a level surface.
* :func:`~fatiando.gravmag.transform.reduce_to_pole`: Reduce the total field
  magnetic anomaly to the pole.
* :func:`~fatiando.gravmag.transform.tga`: Calculate the amplitude of the
  total gradient (also called the analytic signal)
* :func:`~fatiando.gravmag.transform.tilt`: Calculates the tilt angle
* :func:`~fatiando.gravmag.transform.power_density_spectra`: Calculates
  the Power Density Spectra of a gridded potential field data.
* :func:`~fatiando.gravmag.transform.radial_average`: Calculates the
  the radial average of a Power Density Spectra using concentring rings.

**Derivatives**

* :func:`~fatiando.gravmag.transform.derivx`: Calculate the n-th order
  derivative of a potential field in the x-direction (North-South)
* :func:`~fatiando.gravmag.transform.derivy`: Calculate the n-th order
  derivative of a potential field in the y-direction (East-West)
* :func:`~fatiando.gravmag.transform.derivz`: Calculate the n-th order
  derivative of a potential field in the z-direction

----

"""
from __future__ import division, absolute_import
import warnings
import numpy

from .. import utils


def reduce_to_pole(x, y, data, shape, inc, dec, sinc, sdec):
    r"""
    Reduce total field magnetic anomaly data to the pole.

    The reduction to the pole if a phase transformation that can be applied to
    total field magnetic anomaly data. It "simulates" how the data would be if
    **both** the Geomagnetic field and the magnetization of the source were
    vertical (:math:`90^\circ` inclination) (Blakely, 1996).

    This functions performs the reduction in the frequency domain (using the
    FFT). The transform filter is (in the frequency domain):

    .. math::

        RTP(k_x, k_y) = \frac{|k|}{
            a_1 k_x^2 + a_2 k_y^2 + a_3 k_x k_y +
            i|k|(b_1 k_x + b_2 k_y)}

    in which :math:`k_x` and :math:`k_y` are the wave-numbers in the x and y
    directions and

    .. math::

        |k| = \sqrt{k_x^2 + k_y^2} \\
        a_1 = m_z f_z - m_x f_x \\
        a_2 = m_z f_z - m_y f_y \\
        a_3 = -m_y f_x - m_x f_y \\
        b_1 = m_x f_z + m_z f_x \\
        b_2 = m_y f_z + m_z f_y

    :math:`\mathbf{m} = (m_x, m_y, m_z)` is the unit-vector of the total
    magnetization of the source and
    :math:`\mathbf{f} = (f_x, f_y, f_z)` is the unit-vector of the Geomagnetic
    field.

    .. note:: Requires gridded data.

    .. warning::

        The magnetization direction of the anomaly source is crucial to the
        reduction-to-the-pole.
        **Wrong values of *sinc* and *sdec* will lead to a wrong reduction.**

    Parameters:

    * x, y : 1d-arrays
        The x, y, z coordinates of each data point.
    * data : 1d-array
        The total field anomaly data at each point.
    * shape : tuple = (nx, ny)
        The shape of the data grid
    * inc, dec : floats
        The inclination and declination of the inducing Geomagnetic field
    * sinc, sdec : floats
        The inclination and declination of the total magnetization of the
        anomaly source. The total magnetization is the vector sum of the
        induced and remanent magnetization. If there is only induced
        magnetization, use the *inc* and *dec* of the Geomagnetic field.

    Returns:

    * rtp : 1d-array
        The data reduced to the pole.

    References:

    Blakely, R. J. (1996), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

    """
    fx, fy, fz = utils.ang2vec(1, inc, dec)
    if sinc is None or sdec is None:
        mx, my, mz = fx, fy, fz
    else:
        mx, my, mz = utils.ang2vec(1, sinc, sdec)
    kx, ky = [k for k in _fftfreqs(x, y, shape, shape)]
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
    ft_pole = rtp*numpy.fft.fft2(numpy.reshape(data, shape))
    return numpy.real(numpy.fft.ifft2(ft_pole)).ravel()


def upcontinue(x, y, data, shape, height):
    r"""
    Upward continuation of potential field data.

    Calculates the continuation through the Fast Fourier Transform in the
    wavenumber domain (Blakely, 1996):

    .. math::

        F\{h_{up}\} = F\{h\} e^{-\Delta z |k|}

    and then transformed back to the space domain. :math:`h_{up}` is the upward
    continue data, :math:`\Delta z` is the height increase, :math:`F` denotes
    the Fourier Transform,  and :math:`|k|` is the wavenumber modulus.

    .. note:: Requires gridded data.

    .. note:: x, y, z and height should be in meters.

    .. note::

        It is not possible to get the FFT of a masked grid. The default
        :func:`fatiando.gridder.interp` call using minimum curvature will not
        be suitable.  Use ``extrapolate=True`` or ``algorithm='nearest'`` to
        get an unmasked grid.

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid
    * height : float
        The height increase (delta z) in meters.

    Returns:

    * cont : array
        The upward continued data

    References:

    Blakely, R. J. (1996), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

    """
    assert x.shape == y.shape, \
        "x and y arrays must have same shape"
    if height <= 0:
        warnings.warn("Using 'height' <= 0 means downward continuation, " +
                      "which is known to be unstable.")
    nx, ny = shape
    # Pad the array with the edge values to avoid instability
    padded, padx, pady = _pad_data(data, shape)
    kx, ky = _fftfreqs(x, y, shape, padded.shape)
    kz = numpy.sqrt(kx**2 + ky**2)
    upcont_ft = numpy.fft.fft2(padded)*numpy.exp(-height*kz)
    cont = numpy.real(numpy.fft.ifft2(upcont_ft))
    # Remove padding
    cont = cont[padx: padx + nx, pady: pady + ny].ravel()
    return cont


def _upcontinue_space(x, y, data, shape, height):
    """
    Upward continuation using the space-domain formula.

    DEPRECATED. Use the better implementation using FFT. Kept here for
    historical reasons.

    """
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


def tga(x, y, data, shape, method='fd'):
    r"""
    Calculate the total gradient amplitude (TGA).

    This the same as the `3D analytic signal` of Roest et al. (1992), but we
    prefer the newer, more descriptive nomenclature suggested by Reid (2012).

    The TGA is defined as the amplitude of the gradient vector of a potential
    field :math:`T` (e.g. the magnetic total field anomaly):

    .. math::

        TGA = \sqrt{
            \left(\frac{\partial T}{\partial x}\right)^2 +
            \left(\frac{\partial T}{\partial y}\right)^2 +
            \left(\frac{\partial T}{\partial z}\right)^2 }

    .. note:: Requires gridded data.

    .. warning::

        If the data is not in SI units, the derivatives will be in
        strange units and so will the total gradient amplitude! I strongly
        recommend converting the data to SI **before** calculating the
        TGA is you need the gradient in Eotvos (use one of the unit conversion
        functions of :mod:`fatiando.utils`).

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid
    * method : string
        The method used to calculate the horizontal derivatives. Options are:
        ``'fd'`` for finite-difference (more stable) or ``'fft'`` for the Fast
        Fourier Transform. The z derivative is always calculated by FFT.

    Returns:

    * tga : 1D-array
        The amplitude of the total gradient

    References:

    Reid, A. (2012), Forgotten truths, myths and sacred cows of Potential
    Fields Geophysics - II, in SEG Technical Program Expanded Abstracts 2012,
    pp. 1-3, Society of Exploration Geophysicists.

    Roest, W., J. Verhoef, and M. Pilkington (1992), Magnetic interpretation
    using the 3-D analytic signal, GEOPHYSICS, 57(1), 116-125,
    doi:10.1190/1.1443174.

    """
    dx = derivx(x, y, data, shape, method=method)
    dy = derivy(x, y, data, shape, method=method)
    dz = derivz(x, y, data, shape)
    res = numpy.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return res


def tilt(x, y, data, shape, xderiv=None, yderiv=None, zderiv=None):
    r"""
    Calculates the potential field tilt, as defined by Miller and Singh (1994)

    .. math::

        tilt(f) = tan^{-1}\left(
            \frac{
                \frac{\partial T}{\partial z}}{
                \sqrt{\frac{\partial T}{\partial x}^2 +
                      \frac{\partial T}{\partial y}^2}}
            \right)

    When used on magnetic total field anomaly data, works best if the data is
    reduced to the pole.

    It's useful to plot the zero contour line of the tilt to represent possible
    outlines of the source bodies. Use matplotlib's ``pyplot.contour`` or
    ``pyplot.tricontour`` for this.

    .. note::

        Requires gridded data if ``xderiv``, ``yderiv`` and ``zderiv`` are not
        given.

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid. Ignored if *xderiv*, *yderiv* and *zderiv* are
        given.
    * xderiv : 1D-array or None
        Optional. Values of the derivative in the x direction.
        If ``None``, will calculated using the default options of
        :func:`~fatiando.gravmag.transform.derivx`
    * yderiv : 1D-array or None
        Optional. Values of the derivative in the y direction.
        If ``None``, will calculated using the default options of
        :func:`~fatiando.gravmag.transform.derivy`
    * zderiv : 1D-array or None
        Optional. Values of the derivative in the z direction.
        If ``None``, will calculated using the default options of
        :func:`~fatiando.gravmag.transform.derivz`

    Returns:

    * tilt : 1D-array
        The tilt angle of the total field in radians.

    References:

    Miller, Hugh G, and Vijay Singh. 1994. "Potential Field Tilt --- a New
    Concept for Location of Potential Field Sources."
    Journal of Applied Geophysics 32 (2--3): 213-17.
    doi:10.1016/0926-9851(94)90022-1.

    """
    if xderiv is None:
        xderiv = derivx(x, y, data, shape)
    if yderiv is None:
        yderiv = derivy(x, y, data, shape)
    if zderiv is None:
        zderiv = derivz(x, y, data, shape)
    horiz_deriv = numpy.sqrt(xderiv**2 + yderiv**2)
    tilt = numpy.arctan2(zderiv, horiz_deriv)
    return tilt


def derivx(x, y, data, shape, order=1, method='fd'):
    """
    Calculate the derivative of a potential field in the x direction.

    .. note:: Requires gridded data.

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
    * shape : tuple = (nx, ny)
        The shape of the grid
    * order : int
        The order of the derivative
    * method : string
        The method used to calculate the derivatives. Options are:
        ``'fd'`` for central finite-differences (more stable) or ``'fft'``
        for the Fast Fourier Transform.

    Returns:

    * deriv : 1D-array
        The derivative

    """
    nx, ny = shape
    assert method in ['fft', 'fd'], \
        'Invalid method "{}".'.format(method)
    if method == 'fft':
        # Pad the array with the edge values to avoid instability
        padded, padx, pady = _pad_data(data, shape)
        kx, _ = _fftfreqs(x, y, shape, padded.shape)
        deriv_ft = numpy.fft.fft2(padded)*(kx*1j)**order
        deriv_pad = numpy.real(numpy.fft.ifft2(deriv_ft))
        # Remove padding from derivative
        deriv = deriv_pad[padx: padx + nx, pady: pady + ny]
    elif method == 'fd':
        datamat = data.reshape(shape)
        dx = (x.max() - x.min())/(nx - 1)
        deriv = numpy.empty_like(datamat)
        deriv[1:-1, :] = (datamat[2:, :] - datamat[:-2, :])/(2*dx)
        deriv[0, :] = deriv[1, :]
        deriv[-1, :] = deriv[-2, :]
        if order > 1:
            deriv = derivx(x, y, deriv, shape, order=order - 1, method='fd')
    return deriv.ravel()


def derivy(x, y, data, shape, order=1, method='fd'):
    """
    Calculate the derivative of a potential field in the y direction.

    .. note:: Requires gridded data.

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
    * shape : tuple = (nx, ny)
        The shape of the grid
    * order : int
        The order of the derivative
    * method : string
        The method used to calculate the derivatives. Options are:
        ``'fd'`` for central finite-differences (more stable) or ``'fft'``
        for the Fast Fourier Transform.

    Returns:

    * deriv : 1D-array
        The derivative

    """
    nx, ny = shape
    assert method in ['fft', 'fd'], \
        'Invalid method "{}".'.format(method)
    if method == 'fft':
        # Pad the array with the edge values to avoid instability
        padded, padx, pady = _pad_data(data, shape)
        _, ky = _fftfreqs(x, y, shape, padded.shape)
        deriv_ft = numpy.fft.fft2(padded)*(ky*1j)**order
        deriv_pad = numpy.real(numpy.fft.ifft2(deriv_ft))
        # Remove padding from derivative
        deriv = deriv_pad[padx: padx + nx, pady: pady + ny]
    elif method == 'fd':
        datamat = data.reshape(shape)
        dy = (y.max() - y.min())/(ny - 1)
        deriv = numpy.empty_like(datamat)
        deriv[:, 1:-1] = (datamat[:, 2:] - datamat[:, :-2])/(2*dy)
        deriv[:, 0] = deriv[:, 1]
        deriv[:, -1] = deriv[:, -2]
        if order > 1:
            deriv = derivy(x, y, deriv, shape, order=order - 1, method='fd')
    return deriv.ravel()


def derivz(x, y, data, shape, order=1, method='fft'):
    """
    Calculate the derivative of a potential field in the z direction.

    .. note:: Requires gridded data.

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
    * shape : tuple = (nx, ny)
        The shape of the grid
    * order : int
        The order of the derivative
    * method : string
        The method used to calculate the derivatives. Options are:
        ``'fft'`` for the Fast Fourier Transform.

    Returns:

    * deriv : 1D-array
        The derivative

    """
    assert method == 'fft', \
        "Invalid method '{}'".format(method)
    nx, ny = shape
    # Pad the array with the edge values to avoid instability
    padded, padx, pady = _pad_data(data, shape)
    kx, ky = _fftfreqs(x, y, shape, padded.shape)
    deriv_ft = numpy.fft.fft2(padded)*numpy.sqrt(kx**2 + ky**2)**order
    deriv = numpy.real(numpy.fft.ifft2(deriv_ft))
    # Remove padding from derivative
    return deriv[padx: padx + nx, pady: pady + ny].ravel()


def power_density_spectra(x, y, data, shape):
    r"""
    Calculates the Power Density Spectra of a 2D gridded potential field
    through the FFT:

    .. math::

        \Phi_{\Delta T}(k_x, k_y) = | F\left{\Delta T \right}(k_x, k_y) |^2

    .. note:: Requires gridded data.

    .. note:: x, y, z and height should be in meters.

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid

    Returns:

    * kx, ky : 2D-arrays
        The wavenumbers of each Power Density Spectra point
    * pds : 2D-array
        The Power Density Spectra of the data
    """
    kx, ky = _fftfreqs(x, y, shape, shape)
    pds = abs(numpy.fft.fft2(numpy.reshape(data, shape)))**2
    return kx, ky, pds


def radial_average_spectrum(kx, ky, pds, max_radius=None, ring_width=None):
    r"""
    Calculates the average of the Power Density Spectra points that falls
    inside concentric rings built around the origin of the wavenumber
    coordinate system with constant width.

    The width of the rings and the inner radius of the biggest ring can be
    changed by setting the optional parameters ring_width and max_radius,
    respectively.

    .. note:: To calculate the radially averaged power density spectra
              use the outputs of the function power_density_spectra as
              input of this one.

    Parameters:

    * kx, ky : 2D-arrays
        The wavenumbers arrays in the `x` and `y` directions
    * pds : 2D-array
        The Power Density Spectra
    * max_radius : float (optional)
        Inner radius of the biggest ring.
        By default it's set as the minimum of kx.max() and ky.max().
        Making it smaller leaves points outside of the averaging,
        and making it bigger includes points nearer to the boundaries.
    * ring_width : float (optional)
        Width of the rings.
        By default it's set as the largest value of :math:`\Delta k_x` and
        :math:`\Delta k_y`, being them the equidistances of the kx and ky
        arrays.
        Making it bigger gives more populated averages, and
        making it smaller lowers the ammount of points per ring
        (use it carefully).

    Returns:

    * k_radial : 1D-array
        Wavenumbers of each Radially Averaged Power Spectrum point.
        Also, the inner radius of the rings.
    * pds_radial : 1D array
        Radially Averaged Power Spectrum
    """
    nx, ny = pds.shape
    if max_radius is None:
        max_radius = min(kx.max(), ky.max())
    if ring_width is None:
        ring_width = max(numpy.unique(kx)[numpy.unique(kx) > 0][0],
                         numpy.unique(ky)[numpy.unique(ky) > 0][0])
    k = numpy.sqrt(kx**2 + ky**2)
    pds_radial = []
    k_radial = []
    radius_i = -1
    while True:
        radius_i += 1
        if radius_i*ring_width > max_radius:
            break
        else:
            if radius_i == 0:
                inside = k <= 0.5*ring_width
            else:
                inside = numpy.logical_and(k > (radius_i - 0.5)*ring_width,
                                           k <= (radius_i + 0.5)*ring_width)
            pds_radial.append(pds[inside].mean())
            k_radial.append(radius_i*ring_width)
    return numpy.array(k_radial), numpy.array(pds_radial)


def _pad_data(data, shape):
    n = _nextpow2(numpy.max(shape))
    nx, ny = shape
    padx = (n - nx)//2
    pady = (n - ny)//2
    padded = numpy.pad(data.reshape(shape), ((padx, padx), (pady, pady)),
                       mode='edge')
    return padded, padx, pady


def _nextpow2(i):
    buf = numpy.ceil(numpy.log(i)/numpy.log(2))
    return int(2**buf)


def _fftfreqs(x, y, shape, padshape):
    """
    Get two 2D-arrays with the wave numbers in the x and y directions.
    """
    nx, ny = shape
    dx = (x.max() - x.min())/(nx - 1)
    fx = 2*numpy.pi*numpy.fft.fftfreq(padshape[0], dx)
    dy = (y.max() - y.min())/(ny - 1)
    fy = 2*numpy.pi*numpy.fft.fftfreq(padshape[1], dy)
    return numpy.meshgrid(fy, fx)[::-1]
