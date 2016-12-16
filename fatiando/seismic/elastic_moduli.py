# coding: utf-8
"""
Defines conversions between elastic moduli and Vp, Vs, and density.
"""
from __future__ import division, unicode_literals


def lame_lambda(vp, vs, density):
    r"""
    Calculate Lamé's first parameter :math:`\lambda`.

    Defined in terms of the P and S wave velocities
    (:math:`v_P` and :math:`v_S`) and the density (:math:`\rho`).

    .. math::

        \lambda = v_P^2 \rho - 2v_S^2 \rho

    Units must be consistent. For example, if *vp* and *vs* are in m/s, the
    density must be kg/m³ (or g/m³, etc).

    Parameters:

    * vp : float or array
        The P wave velocity
    * vs : float or array
        The S wave velocity
    * density : float or array
        The density

    Returns:

    * lambda : float or array
        The Lamé parameter

    Examples::

        >>> print(lame_lambda(vp=2350, vs=1125, density=2500))
        7478125000
        >>> import numpy as np
        >>> vp = np.array([2000., 3000.])
        >>> vs = np.array([1000., 1700.])
        >>> density = np.array([2700., 3100.])
        >>> print(lame_lambda(vp, vs, density))
        [  5.40000000e+09   9.98200000e+09]

    """
    lamb = density*vp**2 - 2*density*vs**2
    return lamb


def lame_mu(vs, density):
    r"""
    Calculate Lamé's second parameter :math:`\mu` (the shear modulus).

    Defined in terms of the S wave velocity (:math:`v_S`) and the density
    (:math:`\rho`).

    .. math::

        \mu = v_S^2 \rho

    Units must be consistent. For example, if *vs* is in m/s, the density must
    be kg/m³ (or g/m³, etc).

    Parameters:

    * vs : float or array
        The S wave velocity
    * density : float or array
        The density

    Returns:

    * mu : float or array
        The Lamé parameter

    Examples::

        >>> print(lame_mu(vs=1125, density=2500))
        3164062500
        >>> import numpy as np
        >>> vs = np.array([1000., 1700.])
        >>> density = np.array([2700., 3100.])
        >>> print(lame_mu(vs, density))
        [  2.70000000e+09   8.95900000e+09]

    """
    mu = density*vs**2
    return mu
