"""
Euler deconvolution methods for potential fields.

* :func:`~fatiando.gravmag.euler.expanding_window`: Run a given solver for the Euler
  deconvolution on an expanding window and return the best estimate
* :func:`~fatiando.gravmag.euler.classic`: The classic solution to Euler's equation
  for potential fields

----

"""
import time

import numpy

import fatiando.logger
import fatiando.gridder
from fatiando import utils

log = fatiando.logger.dummy('fatiando.gravmag.euler')


def expanding_window(xp, yp, zp, field, xderiv, yderiv, zderiv, index,
    euler, center, minsize, maxsize, nwindows=20):
    """
    Perform the Euler deconvolution on windows of growing size and return the
    best estimate.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the observation points of the **whole**

        data set
    * field : array
        The potential field measured at the observation points of the **whole**
        data set
    * xderiv, yderiv, zderiv : arrays
        The x-, y-, and z-derivatives of the potential field measured
        (calculated) at the observation points of the **whole** data set
    * index : float
        The structural index of the source
    * euler : function
        The Euler deconvolution solver function (like
        :func:`fatiando.gravmag.euler.classic`)
    * center : [x, y]
        The coordinates of the center of the expanding window
    * minsize, maxsize : floats
        The minimum and maximum size of the expanding window
    * nwindows : int
        Number of windows between minsize and maxsize

    Returns:

    * results : dict
        The best results in a dictionary::

            {'point':[x, y, z], # The estimated coordinates of source
             'baselevel':baselevel, # The estimated baselevel
             'mean error':mean_error, # The mean error of the estimated location
             'uncertainty':[sx, sy, sz] # The uncertainty in x, y, and z
            }


    """
    x, y = center
    best = None
    for size in numpy.linspace(minsize, maxsize, nwindows):
        area = [x - 0.5*size, x + 0.5*size, y - 0.5*size, y + 0.5*size]
        subx, suby, scalars = fatiando.gridder.cut(xp, yp,
            [zp, field, xderiv, yderiv, zderiv], area)
        subz, subfield, subxderiv, subyderiv, subzderiv  = scalars
        results = euler(subx, suby, subz, subfield, subxderiv, subyderiv,
            subzderiv, index)
        if best is None or results['mean error'] < best['mean error']:
            best = results
    return best

def classic(xp, yp, zp, field, xderiv, yderiv, zderiv, index):
    """
    Classic 3D Euler deconvolution of potential field data.

    Works on any potential field that satisfies Euler's homogeneity equation.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the observation points
    * field : array
        The potential field measured at the observation points
    * xderiv, yderiv, zderiv : arrays
        The x-, y-, and z-derivatives of the potential field measured
        (calculated) at the observation points
    * index : float
        The structural index of the source

    Returns:

    * results : dict
        The results in a dictionary::

            {'point':[x, y, z], # The estimated coordinates of source
             'baselevel':baselevel, # The estimated baselevel
             'mean error':mean_error, # The mean error of the estimated location
             'uncertainty':[sx, sy, sz] # The uncertainty in x, y, and z
            }


    .. note:: The uncertainty estimate is not very reliable.

    """
    log.info("3D Euler deconvolution using the classic formulation:")
    if (len(xp) != len(yp) != len(zp) != len(field) != len(xderiv)
        != len(yderiv) != len(zderiv)):
        raise ValueError("xp, yp, zp, field, xderiv, yderiv, zderiv need to " +
            "have the same number of elements")
    if index < 0:
        raise ValueError("Invalid structural index '%g'. Should be >= 0"
            % (index))
    log.info("  number of data: %d" % (len(field)*4))
    log.info("  structural index: %g" % (index))
    tstart = time.clock()
    # Using (xp - x) not (x - xp)
    jacobian = numpy.array([-xderiv, -yderiv, -zderiv,
        -index*numpy.ones_like(field)]).T
    data = -xp*xderiv - yp*yderiv - zp*zderiv - index*field
    GGT_inv = numpy.linalg.inv(numpy.dot(jacobian.T, jacobian))
    estimate = numpy.dot(GGT_inv, numpy.dot(jacobian.T, data))
    residuals = data - numpy.dot(jacobian, estimate)
    variance = numpy.sum(residuals**2)/(len(data) - 4) # 4 parameters
    covar = variance*GGT_inv
    uncertainty = numpy.sqrt(numpy.diagonal(covar)[0:3])
    mean_error = numpy.sqrt(numpy.sum(uncertainty**2))
    x, y, z, base = estimate
    log.info("  time it took: %s" % (utils.sec2hms(time.clock() - tstart)))
    log.info("  estimated base level: %g" % (base))
    results = {'point':[x, y, z], 'baselevel':base, 'mean error':mean_error,
               'uncertainty':uncertainty}
    return results
