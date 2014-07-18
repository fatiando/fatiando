"""
This is a Python + Numpy implementation of the potential field effects of
right rectangular prisms.
"""
import numpy
from numpy import sqrt, log, arctan2

from ..constants import SI2EOTVOS, SI2MGAL, G, CM, T2NT
from .. import utils


def potential(xp, yp, zp, prisms, dens=None):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
                    kernel = (x[i]*y[j]*log(z[k] + r)
                              + y[j]*z[k]*log(x[i] + r)
                              + x[i]*z[k]*log(y[j] + r)
                              - 0.5*x[i]**2 *
                              arctan2(z[k]*y[j], x[i]*r)
                              - 0.5*y[j]**2 *
                              arctan2(z[k]*x[i], y[j]*r)
                              - 0.5*z[k]**2*arctan2(x[i]*y[j], z[k]*r))
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant
    res *= G
    return res


def gx(xp, yp, zp, prisms, dens=None):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
                    kernel = -(y[j]*log(z[k] + r)
                               + z[k]*log(y[j] + r)
                               - x[i]*arctan2(z[k]*y[j], x[i]*r))
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units
    res *= G*SI2MGAL
    return res


def gy(xp, yp, zp, prisms, dens=None):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
                    kernel = -(z[k]*log(x[i] + r)
                               + x[i]*log(z[k] + r)
                               - y[j]*arctan2(x[i]*z[k], y[j]*r))
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units
    res *= G*SI2MGAL
    return res


def gz(xp, yp, zp, prisms, dens=None):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
                    kernel = -(x[i]*log(y[j] + r)
                               + y[j]*log(x[i] + r)
                               - z[k]*arctan2(x[i]*y[j], z[k]*r))
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to mGal units
    res *= G*SI2MGAL
    return res


def gxx(xp, yp, zp, prisms, dens=None):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
                    kernel = -arctan2(z[k]*y[j], x[i]*r)
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res


def gxy(xp, yp, zp, prisms, dens=None):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
                    kernel = log(z[k] + r)
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res


def gxz(xp, yp, zp, prisms, dens=None):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
                    kernel = log(y[j] + r)
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res


def gyy(xp, yp, zp, prisms, dens=None):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
                    kernel = -arctan2(z[k]*x[i], y[j]*r)
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res


def gyz(xp, yp, zp, prisms, dens=None):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
                    kernel = log(x[i] + r)
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res


def gzz(xp, yp, zp, prisms, dens=None):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
                    kernel = -arctan2(x[i]*y[j], z[k]*r)
                    res += ((-1.)**(i + j + k))*kernel*density
    # Now all that is left is to multiply res by the gravitational constant and
    # convert it to Eotvos units
    res *= G*SI2EOTVOS
    return res


def tf(xp, yp, zp, prisms, inc, dec, pmag=None):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
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
                        log((r - x[i]) / (r + x[i]))
                        + 0.5*(mx*fz + mz*fx) *
                        log((r - y[j]) / (r + y[j]))
                        - (mx*fy + my*fx)*log(r + z[k])
                        - mx*fx*arctan2(xy, x_sqr + zr + z_sqr)
                        - my*fy*arctan2(xy, r_sqr + zr - x_sqr)
                        + mz*fz*arctan2(xy, zr))
    res *= CM*T2NT
    return res


def bx(xp, yp, zp, prisms):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    bx = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = prism.props['magnetization']
        v1 = kernelxx(xp, yp, zp, prism)
        v2 = kernelxy(xp, yp, zp, prism)
        v3 = kernelxz(xp, yp, zp, prism)
        bx += (v1*mx + v2*my + v3*mz)
    bx *= CM*T2NT
    return bx


def by(xp, yp, zp, prisms):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    by = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = prism.props['magnetization']
        v2 = kernelxy(xp, yp, zp, prism)
        v4 = kernelyy(xp, yp, zp, prism)
        v5 = kernelyz(xp, yp, zp, prism)
        by += (v2*mx + v4*my + v5*mz)
    by *= CM*T2NT
    return by


def bz(xp, yp, zp, prisms):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    bz = numpy.zeros_like(xp)
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = prism.props['magnetization']
        v3 = kernelxz(xp, yp, zp, prism)
        v5 = kernelyz(xp, yp, zp, prism)
        v6 = kernelzz(xp, yp, zp, prism)
        bz += (v3*mx + v5*my + v6*mz)
    bz *= CM*T2NT
    return bz


def kernelxx(xp, yp, zp, prism):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    x1, x2 = prism.x1, prism.x2
    y1, y2 = prism.y1, prism.y2
    z1, z2 = prism.z1, prism.z2
    # Calculate the effect of the prism
    X1 = xp - x1
    X2 = xp - x2
    Y1 = yp - y1
    Y2 = yp - y2
    Z1 = zp - z1
    Z2 = zp - z2
    R111 = numpy.sqrt(X1**2 + Y1**2 + Z1**2)
    R112 = numpy.sqrt(X1**2 + Y1**2 + Z2**2)
    R121 = numpy.sqrt(X1**2 + Y2**2 + Z1**2)
    R122 = numpy.sqrt(X1**2 + Y2**2 + Z2**2)
    R211 = numpy.sqrt(X2**2 + Y1**2 + Z1**2)
    R212 = numpy.sqrt(X2**2 + Y1**2 + Z2**2)
    R221 = numpy.sqrt(X2**2 + Y2**2 + Z1**2)
    R222 = numpy.sqrt(X2**2 + Y2**2 + Z2**2)
    res += -numpy.arctan2((Y1*Z1), (X1*R111))
    res += numpy.arctan2((Y1*Z2), (X1*R112))
    res += numpy.arctan2((Y2*Z1), (X1*R121))
    res += -numpy.arctan2((Y2*Z2), (X1*R122))
    res += numpy.arctan2((Y1*Z1), (X2*R211))
    res += -numpy.arctan2((Y1*Z2), (X2*R212))
    res += -numpy.arctan2((Y2*Z1), (X2*R221))
    res += numpy.arctan2((Y2*Z2), (X2*R222))
    return res


def kernelyy(xp, yp, zp, prism):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    x1, x2 = prism.x1, prism.x2
    y1, y2 = prism.y1, prism.y2
    z1, z2 = prism.z1, prism.z2
    # Calculate the effect of the prism
    X1 = xp - x1
    X2 = xp - x2
    Y1 = yp - y1
    Y2 = yp - y2
    Z1 = zp - z1
    Z2 = zp - z2
    R111 = numpy.sqrt(X1**2 + Y1**2 + Z1**2)
    R112 = numpy.sqrt(X1**2 + Y1**2 + Z2**2)
    R121 = numpy.sqrt(X1**2 + Y2**2 + Z1**2)
    R122 = numpy.sqrt(X1**2 + Y2**2 + Z2**2)
    R211 = numpy.sqrt(X2**2 + Y1**2 + Z1**2)
    R212 = numpy.sqrt(X2**2 + Y1**2 + Z2**2)
    R221 = numpy.sqrt(X2**2 + Y2**2 + Z1**2)
    R222 = numpy.sqrt(X2**2 + Y2**2 + Z2**2)
    res += -numpy.arctan2((X1*Z1), (Y1*R111))
    res += numpy.arctan2((X1*Z2), (Y1*R112))
    res += numpy.arctan2((X1*Z1), (Y2*R121))
    res += -numpy.arctan2((X1*Z2), (Y2*R122))
    res += numpy.arctan2((X2*Z1), (Y1*R211))
    res += -numpy.arctan2((X2*Z2), (Y1*R212))
    res += -numpy.arctan2((X2*Z1), (Y2*R221))
    res += numpy.arctan2((X2*Z2), (Y2*R222))
    return res


def kernelzz(xp, yp, zp, prism):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    x1, x2 = prism.x1, prism.x2
    y1, y2 = prism.y1, prism.y2
    z1, z2 = prism.z1, prism.z2
    # Calculate the effect of the prism
    X1 = xp - x1
    X2 = xp - x2
    Y1 = yp - y1
    Y2 = yp - y2
    Z1 = zp - z1
    Z2 = zp - z2
    R111 = numpy.sqrt(X1**2 + Y1**2 + Z1**2)
    R112 = numpy.sqrt(X1**2 + Y1**2 + Z2**2)
    R121 = numpy.sqrt(X1**2 + Y2**2 + Z1**2)
    R122 = numpy.sqrt(X1**2 + Y2**2 + Z2**2)
    R211 = numpy.sqrt(X2**2 + Y1**2 + Z1**2)
    R212 = numpy.sqrt(X2**2 + Y1**2 + Z2**2)
    R221 = numpy.sqrt(X2**2 + Y2**2 + Z1**2)
    R222 = numpy.sqrt(X2**2 + Y2**2 + Z2**2)
    res += -numpy.arctan2((X1*Y1), (Z1*R111))
    res += numpy.arctan2((X1*Y1), (Z2*R112))
    res += numpy.arctan2((X1*Y2), (Z1*R121))
    res += -numpy.arctan2((X1*Y2), (Z2*R122))
    res += numpy.arctan2((X2*Y1), (Z1*R211))
    res += -numpy.arctan2((X2*Y1), (Z2*R212))
    res += -numpy.arctan2((X2*Y2), (Z1*R221))
    res += numpy.arctan2((X2*Y2), (Z2*R222))
    return res


def kernelxy(xp, yp, zp, prism):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    x1, x2 = prism.x1, prism.x2
    y1, y2 = prism.y1, prism.y2
    z1, z2 = prism.z1, prism.z2
    # Calculate the effect of the prism
    X1 = xp - x1
    X2 = xp - x2
    Y1 = yp - y1
    Y2 = yp - y2
    Z1 = zp - z1
    Z2 = zp - z2
    R111 = numpy.sqrt(X1**2 + Y1**2 + Z1**2)
    R112 = numpy.sqrt(X1**2 + Y1**2 + Z2**2)
    R121 = numpy.sqrt(X1**2 + Y2**2 + Z1**2)
    R122 = numpy.sqrt(X1**2 + Y2**2 + Z2**2)
    R211 = numpy.sqrt(X2**2 + Y1**2 + Z1**2)
    R212 = numpy.sqrt(X2**2 + Y1**2 + Z2**2)
    R221 = numpy.sqrt(X2**2 + Y2**2 + Z1**2)
    R222 = numpy.sqrt(X2**2 + Y2**2 + Z2**2)
    res += -numpy.log((Z1 + R111))
    res += numpy.log((Z2 + R112))
    res += numpy.log((Z1 + R121))
    res += -numpy.log((Z2 + R122))
    res += numpy.log((Z1 + R211))
    res += -numpy.log((Z2 + R212))
    res += -numpy.log((Z1 + R221))
    res += numpy.log((Z2 + R222))
    return -res


def kernelxz(xp, yp, zp, prism):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    x1, x2 = prism.x1, prism.x2
    y1, y2 = prism.y1, prism.y2
    z1, z2 = prism.z1, prism.z2
    # Calculate the effect of the prism
    X1 = xp - x1
    X2 = xp - x2
    Y1 = yp - y1
    Y2 = yp - y2
    Z1 = zp - z1
    Z2 = zp - z2
    R111 = numpy.sqrt(X1**2 + Y1**2 + Z1**2)
    R112 = numpy.sqrt(X1**2 + Y1**2 + Z2**2)
    R121 = numpy.sqrt(X1**2 + Y2**2 + Z1**2)
    R122 = numpy.sqrt(X1**2 + Y2**2 + Z2**2)
    R211 = numpy.sqrt(X2**2 + Y1**2 + Z1**2)
    R212 = numpy.sqrt(X2**2 + Y1**2 + Z2**2)
    R221 = numpy.sqrt(X2**2 + Y2**2 + Z1**2)
    R222 = numpy.sqrt(X2**2 + Y2**2 + Z2**2)
    res += -numpy.log((Y1 + R111))
    res += numpy.log((Y1 + R112))
    res += numpy.log((Y2 + R121))
    res += -numpy.log((Y2 + R122))
    res += numpy.log((Y1 + R211))
    res += -numpy.log((Y1 + R212))
    res += -numpy.log((Y2 + R221))
    res += numpy.log((Y2 + R222))
    return -res


def kernelyz(xp, yp, zp, prism):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = numpy.zeros(len(xp), dtype=numpy.float)
    x1, x2 = prism.x1, prism.x2
    y1, y2 = prism.y1, prism.y2
    z1, z2 = prism.z1, prism.z2
    # Calculate the effect of the prism
    X1 = xp - x1
    X2 = xp - x2
    Y1 = yp - y1
    Y2 = yp - y2
    Z1 = zp - z1
    Z2 = zp - z2
    R111 = numpy.sqrt(X1**2 + Y1**2 + Z1**2)
    R112 = numpy.sqrt(X1**2 + Y1**2 + Z2**2)
    R121 = numpy.sqrt(X1**2 + Y2**2 + Z1**2)
    R122 = numpy.sqrt(X1**2 + Y2**2 + Z2**2)
    R211 = numpy.sqrt(X2**2 + Y1**2 + Z1**2)
    R212 = numpy.sqrt(X2**2 + Y1**2 + Z2**2)
    R221 = numpy.sqrt(X2**2 + Y2**2 + Z1**2)
    R222 = numpy.sqrt(X2**2 + Y2**2 + Z2**2)
    res += -numpy.log((X1 + R111))
    res += numpy.log((X1 + R112))
    res += numpy.log((X1 + R121))
    res += -numpy.log((X1 + R122))
    res += numpy.log((X2 + R211))
    res += -numpy.log((X2 + R212))
    res += -numpy.log((X2 + R221))
    res += numpy.log((X2 + R222))
    return -res
