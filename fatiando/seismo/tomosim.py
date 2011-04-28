# Copyright 2010 The Fatiando a Terra Development Team
#
# This file is part of Fatiando a Terra.
#
# Fatiando a Terra is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Fatiando a Terra is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Fatiando a Terra.  If not, see <http://www.gnu.org/licenses/>.
"""
Very simplified 2D travel time tomography.

Uses straight seismic rays, i.e. does not consider reflection or refraction.

Functions:

* :func:`fatiando.seismo.simpletom.clear`
    Erase garbage from previous inversions

* :func:`fatiando.inv.simpletom.set_bounds`
    Set bounds on the velocity values.

* :func:`fatiando.inv.simpletom.solve`
    Solve the tomography problem for a given data set and model mesh

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 26-Apr-2011'


import logging
import time

import numpy
import scipy.sparse
import scipy.sparse.linalg


import fatiando
from fatiando.seismo import traveltime
from fatiando.inv import reg, gsolvers


log = logging.getLogger('fatiando.seismo.tomosim')
log.addHandler(fatiando.default_log_handler)


def passcells(i, srcx, srcy, recx, recy, cells):
    """
    Determine through which cells the ray passes.
    """
    def ttimes(cells):
        for cell in cells:
            x1, x2, y1, y2 = cell['x1'], cell['x2'], cell['y1'], cell['y2']
            if (x2 < min(srcx, recx) or x1 > max(srcx, recx) or
                y2 < min(srcy, recy) or y1 > max(srcy, recy)):
                yield 0
            else:
                yield traveltime.straight2d(1., x1, y2, x2, y2, srcx, srcy,
                                            recx, recy)
    return numpy.array([[i,j,tt] for j, tt in enumerate(ttimes(cells)) if tt != 0]).T.tolist()


def build_jacobian(sources, receivers, cells):
    """
    Build the Jacobian matrix for the inversion.

    Parameters:

    * sources:
        List with (x,y) coordinate pairs for the wave sources.

    * receivers:
        List with (x,y) coordinate pairs for the wave receivers.

    * cells:
        List of cells in the square mesh used in the inversion.

    Returns:

    * Jacobian matrix:
        sparse matrix with the sensibility of each travel time to each cell

    """
    row, col, val = [], [], []
    appendr, appendc, appendv = row.append, col.append, val.append
    for i, pair in enumerate(zip(sources, receivers)):
        src, rec = pair
        for j, cell in enumerate(cells):
            tt = traveltime.straight2d(1., cell['x1'], cell['y1'], cell['x2'],
                                       cell['y2'], src[0], src[1], rec[0],
                                       rec[1])
            if tt != 0:
                appendr(i)
                appendc(j)
                appendv(tt)
    return scipy.sparse.csr_matrix((val,(row,col)), (len(sources),len(cells)))


def solve(data, mesh, damp=0, smooth=0):
    """
    Solve the tomography problem for the velocity values in each cell of a
    discretization mesh.

    Parameters:

    * data
        Travel time data stored in a dictionary (see
        :func:`fatiando.seismo.synthetic.shoot_straight2d`)

    * mesh
        Discretization mesh defining the inversion parameters
        (see :func:`fatiando.mesh.square_mesh`)

    * damp
        Damping regularization parameter (how much damping to apply).
        Must be >= 0

    Return:

    * results in a dictionary:
        {'estimate' : 1D array with the final estimate,
         'residuals' : 1D array with the travel time residuals}

    """
    log.info("TomoSim solver:")
    log.info("  damping    = %g" % (damp))
    log.info("  smoothness = %g" % (smooth))

    # Build the jacobian (sensibility) matrix
    start = time.time()
    jacobian = build_jacobian(data['src'], data['rec'], mesh.ravel())
    log.info("  Built Jacobian matrix (%.2f s)" % (time.time() - start))

    def func(p, jacobian=jacobian):
        "Functional relation between the slowness and the travel-times"
        return jacobian*p

    fdmatrix = reg.fdmatrix2d(*mesh.shape)
    tk1weights = numpy.dot(fdmatrix.T, fdmatrix)
    def reghess(hess, damp=damp, smooth=smooth, tk1weights=tk1weights):
        "Sum the hessian of the regularizing functions"
        return reg.damp_hess(damp,
                    reg.smooth2d_hess(smooth, tk1weights, hess))

    if damp == 0 and smooth == 0:
        regnorm, reggrad, reghess = None, None, None

    # Configure gsolvers to use sparse matrix operations
    gsolvers.dot_product = scipy.sparse.csc_matrix.__mul__
    def linsys_solver(A, x):
        res = scipy.sparse.linalg.cgs(A, x)
        if res[1] > 0:
            log.warning("Conjugate Gradient convergence not achieved")
        if res[1] < 0:
            log.error("Conjugate Gradient illegal input or breakdown")
        return res[0]
    gsolvers.linsys_solver = linsys_solver

    start = time.time()
    results = gsolvers.linlsq(data['traveltime'], func, jacobian, None, reghess)
    log.info("  Time for inversion: %.2f s" % (time.time() - start))
    # The inversion outputs a slowness estimate. Convert it to velocity
    results['estimate'] = 1./results['estimate']
    return results


def isolve(data, mesh, init, damp=0, lmstart=100, lmstep=10, maxsteps=20,
          maxit=100, tol=10**(-5)):
    """
    Iteratively solve the tomography problem for the velocity values in each
    cell of a discretization mesh. Uses the Levemberg-Marquardt algorithm.

    Parameters:

    * data
        Travel time data stored in a dictionary (see
        :func:`fatiando.seismo.synthetic.shoot_straight2d`)

    * mesh
        Discretization mesh defining the inversion parameters
        (see :func:`fatiando.mesh.square_mesh`)

    * init
        1D array with the initial estimate of the velocity values

    * damp
        Damping regularization parameter (how much damping to apply).
        Must be >= 0

    * lmstart
        Initial Marquardt parameter (ie, step size)

    * lmstep
        Factor by which the Marquardt parameter will be reduced with each
        successful step

    * maxsteps
        How many times to try giving a step before exiting

    * maxit
        Maximum number of iterations

    * tol
        Relative tolerance for decreasing the goal function to achieve before
        terminating

    Return:

    * results in a dictionary:
        {'estimate' : 1D array with the final estimate,
         'residuals' : 1D array with the travel time residuals,
         'goal_p_it' : 1D array with the value of the goal function per
                       iteration}

    """
    log.info("TomoSim iterative solver:")
    log.info("  damping    = %g" % (damp))
    log.info("  lmstart    = %g" % (lmstart))
    log.info("  lmstep     = %g" % (lmstep))
    log.info("  maxsteps   = %g" % (maxsteps))
    log.info("  maxit      = %g" % (maxit))
    log.info("  tolerance  = %g" % (tol))

    # Build the jacobian (sensibility) matrix
    start = time.time()
    jacobian = build_jacobian(data['src'], data['rec'], mesh.ravel())
    log.info("  Built Jacobian matrix (%.2f s)" % (time.time() - start))

    def jac(p, jacobian=jacobian):
        "Calculate the Jacobian (sensibility) matrix"
        return jacobian

    def func(p, jacobian=jacobian):
        "Functional relation between the slowness and the travel-times"
        return jacobian*p

    def regnorm(p, damp=damp):
        "Calculate the norm of the regularizing functions"
        return reg.damp_norm(damp, p)

    def reggrad(grad, p, damp=damp):
        "Sum the gradient of the regularizing functions"
        return reg.damp_grad(damp, p, grad)

    def reghess(hess, p=None, damp=damp):
        "Sum the hessian of the regularizing functions"
        return reg.damp_hess(damp, hess)

    if damp == 0:
        regnorm, reggrad, reghess = None, None, None

    # Configure gsolvers to use sparse matrix operations
    gsolvers.dot_product = scipy.sparse.csc_matrix.__mul__
    def linsys_solver(A, x):
        res = scipy.sparse.linalg.cgs(A, x)
        if res[1] > 0:
            log.warning("Conjugate Gradient convergence not achieved")
        if res[1] < 0:
            log.error("Conjugate Gradient illegal input or breakdown")
        return res[0]
    gsolvers.linsys_solver = linsys_solver

    start = time.time()
    # Convert velocity to slowness so that the inversion is linear
    results = gsolvers.marq(data['traveltime'], 1./numpy.array(init), func, jac,
                            lmstart, lmstep, maxsteps, maxit, tol, regnorm,
                            reggrad, reghess)
    log.info("  Time for inversion: %.2f s" % (time.time() - start))

    # The inversion outputs a slowness estimate. Convert it to velocity
    results['estimate'] = 1./results['estimate']

    return results
