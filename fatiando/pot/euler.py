"""
Euler deconvolution methods for potential fields.


"""
import numpy

import fatiando.log

log = fatiando.log.dummy('fatiando.pot.euler')


def classic(xp, yp, zp, field, xderiv, yderiv, zderiv, index):
    """
    Classic 3D Euler deconvolution.

    Returns:

    * [[x, y, z], base]

    """
    # NOTE: use (xp - x) not (x - xp)
    jacobian = numpy.array([-xderiv, -yderiv, -zderiv,
        -index*numpy.ones_like(field)]).T
    data = -xp*xderiv - yp*yderiv - zp*zderiv - index*field
    x, y, z, base = numpy.linalg.solve(
        numpy.dot(jacobian.T, jacobian), numpy.dot(jacobian.T, data))
    return [x, y, z], base
