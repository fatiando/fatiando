from __future__ import division

import numpy as np

from ... import utils
from ...constants import G, SI2EOTVOS, CM, T2NT, SI2MGAL


def safe_atan(y, x):
    """
    Correct the value of the angle returned by arctan2 to match the sign of the
    tangent. Also return 0 instead of 2Pi for 0 tangent.
    """
    res = np.arctan2(y, x)
    res[y == 0] = 0
    res[(y > 0) & (x < 0)] -= np.pi
    res[(y < 0) & (x < 0)] += np.pi
    return res


def safe_log(x, tol=1e-3):
    """
    Return 0 for log(0) because the limits in the formula terms tend to 0
    (see Nagy et al., 2000)
    """
    res = np.log(x)
    res[np.abs(x) < tol] = 0
    return res


def potential(xp, yp, zp, prism, dens=None):
    assert xp.shape == yp.shape == zp.shape, \
        "Input arrays x, y, z must have same shape"
    res = np.zeros_like(xp)
    if dens is None:
        density = prism.props['density']
    else:
        density = dens
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    limits_x = [prism.x2 - xp, prism.x1 - xp]
    limits_y = [prism.y2 - yp, prism.y1 - yp]
    limits_z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k, z in enumerate(limits_z):
        for j, y in enumerate(limits_y):
            for i, x in enumerate(limits_x):
                r = np.sqrt(x*x + y*y + z*z)
                kernel = (x*y*safe_log(z + r)
                          + y*z*safe_log(x + r)
                          + x*z*safe_log(y + r)
                          - (0.5*x*x)*safe_atan(z*y, x*r)
                          - (0.5*y*y)*safe_atan(z*x, y*r)
                          - (0.5*z*z)*safe_atan(x*y, z*r))
                res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant
    res *= G
    return res


def gx(xp, yp, zp, prism, dens=None):
    assert xp.shape == yp.shape == zp.shape, \
        "Input arrays x, y, z must have same shape"
    res = np.zeros_like(xp)
    if dens is None:
        density = prism.props['density']
    else:
        density = dens
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    limits_x = [prism.x2 - xp, prism.x1 - xp]
    limits_y = [prism.y2 - yp, prism.y1 - yp]
    limits_z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k, z in enumerate(limits_z):
        for j, y in enumerate(limits_y):
            for i, x in enumerate(limits_x):
                r = np.sqrt(x*x + y*y + z*z)
                # Minus because Nagy et al (2000) give the formula for the
                # gradient of the potential. Gravity is -grad(V)
                kernel = -(y*safe_log(z + r)
                           + z*safe_log(y + r)
                           - x*safe_atan(z*y, x*r))
                res += ((-1)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant
    res *= G*SI2MGAL
    return res


def gy(xp, yp, zp, prism, dens=None):
    assert xp.shape == yp.shape == zp.shape, \
        "Input arrays x, y, z must have same shape"
    res = np.zeros_like(xp)
    if dens is None:
        density = prism.props['density']
    else:
        density = dens
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    limits_x = [prism.x2 - xp, prism.x1 - xp]
    limits_y = [prism.y2 - yp, prism.y1 - yp]
    limits_z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k, z in enumerate(limits_z):
        for j, y in enumerate(limits_y):
            for i, x in enumerate(limits_x):
                r = np.sqrt(x*x + y*y + z*z)
                # Minus because Nagy et al (2000) give the formula for the
                # gradient of the potential. Gravity is -grad(V)
                kernel = -(z*safe_log(x + r)
                           + x*safe_log(z + r)
                           - y*safe_atan(x*z, y*r))
                res += ((-1)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant
    res *= G*SI2MGAL
    return res


def gz(xp, yp, zp, prism, dens=None):
    assert xp.shape == yp.shape == zp.shape, \
        "Input arrays x, y, z must have same shape"
    res = np.zeros_like(xp)
    if dens is None:
        density = prism.props['density']
    else:
        density = dens
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    limits_x = [prism.x2 - xp, prism.x1 - xp]
    limits_y = [prism.y2 - yp, prism.y1 - yp]
    limits_z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k, z in enumerate(limits_z):
        for j, y in enumerate(limits_y):
            for i, x in enumerate(limits_x):
                r = np.sqrt(x*x + y*y + z*z)
                # Minus because Nagy et al (2000) give the formula for the
                # gradient of the potential. Gravity is -grad(V)
                kernel = -(x*safe_log(y + r)
                           + y*safe_log(x + r)
                           - z*safe_atan(x*y, z*r))
                res += ((-1)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant
    res *= G*SI2MGAL
    return res


def gxx(xp, yp, zp, prism, dens=None):
    if dens is None:
        density = prism.props['density']
    else:
        density = dens
    res = kernelxx(xp, yp, zp, prism)
    # Now all that is left is to multiply res by the gravitational constant
    res *= G*SI2EOTVOS*density
    return res


def kernelxx(xp, yp, zp, prism):
    assert xp.shape == yp.shape == zp.shape, \
        "Input arrays x, y, z must have same shape"
    res = np.zeros_like(xp)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    limits_x = [prism.x2 - xp, prism.x1 - xp]
    limits_y = [prism.y2 - yp, prism.y1 - yp]
    limits_z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k, z in enumerate(limits_z):
        for j, y in enumerate(limits_y):
            for i, x in enumerate(limits_x):
                r = np.sqrt(x*x + y*y + z*z)
                kernel = -safe_atan(z*y, x*r)
                res += ((-1)**(i + j + k))*kernel
    return res


def gxy(xp, yp, zp, prism, dens=None):
    if dens is None:
        density = prism.props['density']
    else:
        density = dens
    res = kernelxy(xp, yp, zp, prism)
    # Now all that is left is to multiply res by the gravitational constant
    res *= G*SI2EOTVOS*density
    return res


def kernelxy(xp, yp, zp, prism):
    assert xp.shape == yp.shape == zp.shape, \
        "Input arrays x, y, z must have same shape"
    res = np.zeros_like(xp)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    limits_x = [prism.x2 - xp, prism.x1 - xp]
    limits_y = [prism.y2 - yp, prism.y1 - yp]
    limits_z = [prism.z2 - zp, prism.z1 - zp]
    # There is a singularity when the computation point is aligned with one of
    # the vertical edges and is below the prism. In such cases, will move the
    # computation point (shift r) by a percentage of the dimensions of the
    # prism.
    # dx = 0.001*abs(prism.x2 - prism.x1)
    # dy = 0.001*abs(prism.y2 - prism.y1)
    # Evaluate the integration limits
    for k, z in enumerate(limits_z):
        for j, y in enumerate(limits_y):
            for i, x in enumerate(limits_x):
                r = np.sqrt(x*x + y*y + z*z)
                # corner = (np.abs(x) < dx) & (np.abs(y) < dy) & (z < 0)
                # if np.any(corner):
                    # r[corner] = np.sqrt(dx**2 + dy**2 + z[corner]**2)
                kernel = safe_log(z + r)
                res += ((-1)**(i + j + k))*kernel
    return res


def gxz(xp, yp, zp, prism, dens=None):
    if dens is None:
        density = prism.props['density']
    else:
        density = dens
    res = kernelxz(xp, yp, zp, prism)
    # Now all that is left is to multiply res by the gravitational constant
    res *= G*SI2EOTVOS*density
    return res


def kernelxz(xp, yp, zp, prism):
    assert xp.shape == yp.shape == zp.shape, \
        "Input arrays x, y, z must have same shape"
    res = np.zeros_like(xp)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    limits_x = [prism.x2 - xp, prism.x1 - xp]
    limits_y = [prism.y2 - yp, prism.y1 - yp]
    limits_z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k, z in enumerate(limits_z):
        for j, y in enumerate(limits_y):
            for i, x in enumerate(limits_x):
                r = np.sqrt(x*x + y*y + z*z)
                kernel = safe_log(y + r)
                res += ((-1)**(i + j + k))*kernel
    return res


def gyy(xp, yp, zp, prism, dens=None):
    if dens is None:
        density = prism.props['density']
    else:
        density = dens
    res = kernelyy(xp, yp, zp, prism)
    # Now all that is left is to multiply res by the gravitational constant
    res *= G*SI2EOTVOS*density
    return res


def kernelyy(xp, yp, zp, prism):
    assert xp.shape == yp.shape == zp.shape, \
        "Input arrays x, y, z must have same shape"
    res = np.zeros_like(xp)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    limits_x = [prism.x2 - xp, prism.x1 - xp]
    limits_y = [prism.y2 - yp, prism.y1 - yp]
    limits_z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k, z in enumerate(limits_z):
        for j, y in enumerate(limits_y):
            for i, x in enumerate(limits_x):
                r = np.sqrt(x*x + y*y + z*z)
                kernel = -safe_atan(z*x, y*r)
                res += ((-1)**(i + j + k))*kernel
    return res


def gyz(xp, yp, zp, prism, dens=None):
    if dens is None:
        density = prism.props['density']
    else:
        density = dens
    res = kernelyz(xp, yp, zp, prism)
    # Now all that is left is to multiply res by the gravitational constant
    res *= G*SI2EOTVOS*density
    return res


def kernelyz(xp, yp, zp, prism):
    assert xp.shape == yp.shape == zp.shape, \
        "Input arrays x, y, z must have same shape"
    res = np.zeros_like(xp)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    limits_x = [prism.x2 - xp, prism.x1 - xp]
    limits_y = [prism.y2 - yp, prism.y1 - yp]
    limits_z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k, z in enumerate(limits_z):
        for j, y in enumerate(limits_y):
            for i, x in enumerate(limits_x):
                r = np.sqrt(x*x + y*y + z*z)
                kernel = safe_log(x + r)
                res += ((-1)**(i + j + k))*kernel
    return res


def gzz(xp, yp, zp, prism, dens=None):
    if dens is None:
        density = prism.props['density']
    else:
        density = dens
    res = kernelzz(xp, yp, zp, prism)
    # Now all that is left is to multiply res by the gravitational constant
    res *= G*SI2EOTVOS*density
    return res


def kernelzz(xp, yp, zp, prism):
    assert xp.shape == yp.shape == zp.shape, \
        "Input arrays x, y, z must have same shape"
    res = np.zeros_like(xp)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    limits_x = [prism.x2 - xp, prism.x1 - xp]
    limits_y = [prism.y2 - yp, prism.y1 - yp]
    limits_z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k, z in enumerate(limits_z):
        for j, y in enumerate(limits_y):
            for i, x in enumerate(limits_x):
                r = np.sqrt(x*x + y*y + z*z)
                kernel = -safe_atan(y*x, z*r)
                res += ((-1)**(i + j + k))*kernel
    return res
