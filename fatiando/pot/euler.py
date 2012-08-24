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

def implicit(xp, yp, zp, field, xderiv, yderiv, zderiv, index, maxit=100,
    tol=10.**(-2)):
    """
    """
    R = len(field)
    N = 4*R
    x, y, z = numpy.mean(xp), numpy.mean(yp), numpy.mean(zp)
    base = 0
    estimate = numpy.array([x, y, z, base], dtype='f')
    pred = numpy.zeros(N, dtype='f')
    pred[:R] = numpy.copy(field)
    pred[R:2*R] = numpy.copy(xderiv)
    pred[2*R:3*R] = numpy.copy(yderiv)
    pred[3*R:4*R] = numpy.copy(zderiv)
    identity = numpy.eye(len(estimate), dtype='f')
    misfit_old = None
    for l in xrange(maxit):
        x, y, z, base = estimate
        pfield = pred[:R]
        pxderiv = pred[R:2*R]
        pyderiv = pred[2*R:3*R]
        pzderiv = pred[3*R:4*R]
        A = numpy.array([-pxderiv, -pyderiv, -pzderiv,
                         -index*numpy.ones_like(field)]).T
        B = numpy.zeros((R, N), dtype='f')
        B[:,:R] = index*numpy.eye(R, dtype='f')
        B[:,R:2*R] = numpy.diag(xp - x)
        B[:,2*R:3*R] = numpy.diag(yp - y)
        B[:,3*R:4*R] = numpy.diag(zp - z)
        w = ((xp - x)*pxderiv + (yp - y)*pyderiv + (zp - z)*pzderiv +
             index*(pfield - base))
        BBT_inv = numpy.linalg.inv(numpy.dot(B, B.T))
        increment = -numpy.linalg.solve(
            numpy.dot(A.T, numpy.dot(BBT_inv, A)),
            numpy.dot(A.T, numpy.dot(BBT_inv, w)))
        residuals = -numpy.dot(B.T,
                        numpy.dot(BBT_inv,
                            numpy.dot(A, increment) + w))
        estimate = estimate + increment
        pred = pred + residuals
        misfit = numpy.linalg.norm(residuals)
        if l != 0 and abs(misfit - misfit_old)/misfit_old <= tol:
            break
        misfit_old = misfit
    log.info("  number of iterations: %d" % (l + 1))
    pfield = pred[:R]
    pxderiv = pred[R:2*R]
    pyderiv = pred[2*R:3*R]
    pzderiv = pred[3*R:4*R]
    x, y, z, base = estimate
    return [x, y, z], base, pfield, pxderiv, pyderiv, pzderiv
        
        
    
    
