"""
This is a Python + Numpy implementation of the potential field effects of
right rectangular prisms. This is used to test the more efficient Cython
version in fatiando.gravmag._prism. Not meant for actual use.
"""
import numpy
from numpy import sqrt, log, arctan2, pi

from ..constants import SI2EOTVOS, SI2MGAL, G, CM, T2NT
from .. import utils


def safe_atan2(y, x):
    """
    Correct the value of the angle returned by arctan2 to match the sign of the
    tangent. Also return 0 instead of 2Pi for 0 tangent.
    """
    res = arctan2(y, x)
    res[y == 0] = 0
    res[(y > 0) & (x < 0)] -= pi
    res[(y < 0) & (x < 0)] += pi
    return res


def safe_log(x):
    """
    Return 0 for log(0) because the limits in the formula terms tend to 0
    (see Nagy et al., 2000)
    """
    res = log(x)
    res[x == 0] = 0
    return res


def potential(xp, yp, zp, prisms, dens=None):
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        # First thing to do is make the computation point P the origin of the
        # coordinate system
        x = [prism.x2 - xp, prism.x1 - xp]
        y = [prism.y2 - yp, prism.y1 - yp]
        z = [prism.z2 - zp, prism.z1 - zp]
        # Evaluate the integration limits
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                    kernel = (x[i]*y[j]*safe_log(z[k] + r)
                              + y[j]*z[k]*safe_log(x[i] + r)
                              + x[i]*z[k]*safe_log(y[j] + r)
                              - 0.5*x[i]**2 *
                              safe_atan2(z[k]*y[j], x[i]*r)
                              - 0.5*y[j]**2 *
                              safe_atan2(z[k]*x[i], y[j]*r)
                              - 0.5*z[k]**2*safe_atan2(x[i]*y[j], z[k]*r))
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant
    res *= G
    return res


def gx(xp, yp, zp, prisms, dens=None):
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        # First thing to do is make the computation point P the origin of the
        # coordinate system
        x = [prism.x2 - xp, prism.x1 - xp]
        y = [prism.y2 - yp, prism.y1 - yp]
        z = [prism.z2 - zp, prism.z1 - zp]
        # Evaluate the integration limits
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                    # Minus because Nagy et al (2000) give the formula for the
                    # gradient of the potential. Gravity is -grad(V)
                    kernel = -(y[j]*safe_log(z[k] + r)
                               + z[k]*safe_log(y[j] + r)
                               - x[i]*safe_atan2(z[k]*y[j], x[i]*r))
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units
    res *= G*SI2MGAL
    return res


def gy(xp, yp, zp, prisms, dens=None):
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        # First thing to do is make the computation point P the origin of the
        # coordinate system
        x = [prism.x2 - xp, prism.x1 - xp]
        y = [prism.y2 - yp, prism.y1 - yp]
        z = [prism.z2 - zp, prism.z1 - zp]
        # Evaluate the integration limits
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                    # Minus because Nagy et al (2000) give the formula for the
                    # gradient of the potential. Gravity is -grad(V)
                    kernel = -(z[k]*safe_log(x[i] + r)
                               + x[i]*safe_log(z[k] + r)
                               - y[j]*safe_atan2(x[i]*z[k], y[j]*r))
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units
    res *= G*SI2MGAL
    return res


def gz(xp, yp, zp, prisms, dens=None):
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        # First thing to do is make the computation point P the origin of the
        # coordinate system
        x = [prism.x2 - xp, prism.x1 - xp]
        y = [prism.y2 - yp, prism.y1 - yp]
        z = [prism.z2 - zp, prism.z1 - zp]
        # Evaluate the integration limits
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                    # Minus because Nagy et al (2000) give the formula for the
                    # gradient of the potential. Gravity is -grad(V)
                    kernel = -(x[i]*safe_log(y[j] + r)
                               + y[j]*safe_log(x[i] + r)
                               - z[k]*safe_atan2(x[i]*y[j], z[k]*r))
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units
    res *= G*SI2MGAL
    return res


def gxx(xp, yp, zp, prisms, dens=None):
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        res += density*kernelxx(xp, yp, zp, prism)
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res


def gxy(xp, yp, zp, prisms, dens=None):
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        res += density*kernelxy(xp, yp, zp, prism)
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res


def gxz(xp, yp, zp, prisms, dens=None):
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        res += density*kernelxz(xp, yp, zp, prism)
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res


def gyy(xp, yp, zp, prisms, dens=None):
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        res += density*kernelyy(xp, yp, zp, prism)
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res


def gyz(xp, yp, zp, prisms, dens=None):
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        res += density*kernelyz(xp, yp, zp, prism)
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res


def gzz(xp, yp, zp, prisms, dens=None):
    res = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        res += density*kernelzz(xp, yp, zp, prism)
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res


def tf(xp, yp, zp, prisms, inc, dec, pmag=None):
    res = numpy.zeros_like(xp)
    # Calculate the 3 components of the unit vector in the direction of the
    # regional field
    fx, fy, fz = utils.dircos(inc, dec)
    if pmag is not None:
        if isinstance(pmag, float) or isinstance(pmag, int):
            pintensity = pmag
            pmx, pmy, pmz = fx, fy, fz
        else:
            pintensity = numpy.linalg.norm(pmag)
            pmx, pmy, pmz = numpy.array(pmag) / pintensity
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props
                             and pmag is None):
            continue
        if pmag is None:
            mag = prism.props['magnetization']
            if isinstance(mag, float) or isinstance(mag, int):
                intensity = mag
                mx, my, mz = fx, fy, fz
            else:
                intensity = numpy.linalg.norm(mag)
                mx, my, mz = numpy.array(mag) / intensity
        else:
            intensity = pintensity
            mx, my, mz = pmx, pmy, pmz
        # First thing to do is make the computation point P the origin of the
        # coordinate system
        x = [prism.x2 - xp, prism.x1 - xp]
        y = [prism.y2 - yp, prism.y1 - yp]
        z = [prism.z2 - zp, prism.z1 - zp]
        # Now calculate the total field anomaly
        for k in range(2):
            intensity *= -1
            z_sqr = z[k]**2
            for j in range(2):
                y_sqr = y[j]**2
                for i in range(2):
                    x_sqr = x[i]**2
                    xy = x[i]*y[j]
                    r_sqr = x_sqr + y_sqr + z_sqr
                    r = sqrt(r_sqr)
                    zr = z[k]*r
                    res += ((-1.)**(i + j))*intensity*(
                        0.5*(my*fz + mz*fy) *
                        safe_log((r - x[i]) / (r + x[i]))
                        + 0.5*(mx*fz + mz*fx) *
                        safe_log((r - y[j]) / (r + y[j]))
                        - (mx*fy + my*fx)*safe_log(r + z[k])
                        - mx*fx*safe_atan2(xy, x_sqr + zr + z_sqr)
                        - my*fy*safe_atan2(xy, r_sqr + zr - x_sqr)
                        + mz*fz*safe_atan2(xy, zr))
    res *= CM*T2NT
    return res


def bx(xp, yp, zp, prisms, pmag=None):
    if pmag is not None:
        mx, my, mz = pmag
    bx = numpy.zeros_like(xp)
    for prism in prisms:
        if (prism is None or
                ('magnetization' not in prism.props and pmag is None)):
            continue
        if pmag is None:
            mx, my, mz = prism.props['magnetization']
        v1 = kernelxx(xp, yp, zp, prism)
        v2 = kernelxy(xp, yp, zp, prism)
        v3 = kernelxz(xp, yp, zp, prism)
        bx += (v1*mx + v2*my + v3*mz)
    bx *= CM*T2NT
    return bx


def by(xp, yp, zp, prisms, pmag=None):
    if pmag is not None:
        mx, my, mz = pmag
    by = numpy.zeros_like(xp)
    for prism in prisms:
        if (prism is None or
                ('magnetization' not in prism.props and pmag is None)):
            continue
        if pmag is None:
            mx, my, mz = prism.props['magnetization']
        v2 = kernelxy(xp, yp, zp, prism)
        v4 = kernelyy(xp, yp, zp, prism)
        v5 = kernelyz(xp, yp, zp, prism)
        by += (v2*mx + v4*my + v5*mz)
    by *= CM*T2NT
    return by


def bz(xp, yp, zp, prisms, pmag=None):
    if pmag is not None:
        mx, my, mz = pmag
    bz = numpy.zeros_like(xp)
    for prism in prisms:
        if (prism is None or
                ('magnetization' not in prism.props and pmag is None)):
            continue
        if pmag is None:
            mx, my, mz = prism.props['magnetization']
        v3 = kernelxz(xp, yp, zp, prism)
        v5 = kernelyz(xp, yp, zp, prism)
        v6 = kernelzz(xp, yp, zp, prism)
        bz += (v3*mx + v5*my + v6*mz)
    bz *= CM*T2NT
    return bz


def kernelxx(xp, yp, zp, prism):
    res = numpy.zeros(len(xp), dtype=numpy.float)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    x = [prism.x2 - xp, prism.x1 - xp]
    y = [prism.y2 - yp, prism.y1 - yp]
    z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                kernel = -safe_atan2(z[k]*y[j], x[i]*r)
                res += ((-1.)**(i + j + k))*kernel
    return res


def kernelyy(xp, yp, zp, prism):
    res = numpy.zeros(len(xp), dtype=numpy.float)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    x = [prism.x2 - xp, prism.x1 - xp]
    y = [prism.y2 - yp, prism.y1 - yp]
    z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                kernel = -safe_atan2(z[k]*x[i], y[j]*r)
                res += ((-1.)**(i + j + k))*kernel
    return res


def kernelzz(xp, yp, zp, prism):
    res = numpy.zeros(len(xp), dtype=numpy.float)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    x = [prism.x2 - xp, prism.x1 - xp]
    y = [prism.y2 - yp, prism.y1 - yp]
    z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                kernel = -safe_atan2(y[j]*x[i], z[k]*r)
                res += ((-1.)**(i + j + k))*kernel
    return res


def kernelxy(xp, yp, zp, prism):
    res = numpy.zeros(len(xp), dtype=numpy.float)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    x = [prism.x2 - xp, prism.x1 - xp]
    y = [prism.y2 - yp, prism.y1 - yp]
    z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                kernel = safe_log(z[k] + r)
                res += ((-1.)**(i + j + k))*kernel
    return res


def kernelxz(xp, yp, zp, prism):
    res = numpy.zeros(len(xp), dtype=numpy.float)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    x = [prism.x2 - xp, prism.x1 - xp]
    y = [prism.y2 - yp, prism.y1 - yp]
    z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                kernel = safe_log(y[j] + r)
                res += ((-1.)**(i + j + k))*kernel
    return res


def kernelyz(xp, yp, zp, prism):
    res = numpy.zeros(len(xp), dtype=numpy.float)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    x = [prism.x2 - xp, prism.x1 - xp]
    y = [prism.y2 - yp, prism.y1 - yp]
    z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                kernel = safe_log(x[i] + r)
                res += ((-1.)**(i + j + k))*kernel
    return res
