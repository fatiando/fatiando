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
A simplified 2D Cartesian tomography problem.
Considers straight (seismic) rays, eg does not consider reflection or refraction

Functions:
  * clear: Erase garbage from previous inversions
  * make_mesh: Make a model space discretization mesh
  * solve: Solve the tomography problem for a given data set and model mesh
  * residuals: Calculate the residuals produced by a given estimate
  * fill_mesh: Fill the 'value' keys of mesh with the values in the estimate
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 29-Apr-2010'


import logging

import numpy

import fatiando
from fatiando.seismo import traveltime
from fatiando.inversion import solvers

log = logging.getLogger('fatiando.inversion.simpletom')  
log.setLevel(logging.DEBUG)
log.addHandler(fatiando.default_log_handler)


# Defined as globals so that they won't have to be re-calculated
_jacobian = None
_mesh = None
_sources = None
_receivers = None


def clear():
    """Erase garbage from previous inversions"""
    
    global _jacobian, _mesh, _sources, _receivers
    
    _jacobian = None
    _mesh = None
    _sources = None
    _receivers = None


def _build_simpletom_jacobian(estimate):
    """Build the Jacobian matrix of the traveltime function."""
    
    assert _mesh is not None, "Can't build simpletom Jacobian. No mesh defined"
    assert _sources is not None, "Can't build simpletom Jacobian." + \
        "No ray sources given"
    assert _receivers is not None, "Can't build simpletom Jacobian." + \
        "No ray receivers given"
    
    global _jacobian
    
    if _jacobian is None:
        
        _jacobian = []
                
        for src, rec in zip(_sources, _receivers):
                        
            line = []
            
            for cell in _mesh.ravel():
                
                value = traveltime.cartesian_straight(1., 
                                                      cell['x1'], cell['y1'], 
                                                      cell['x2'], cell['y2'], 
                                                      src[0], src[1], 
                                                      rec[0], rec[1])
                
                line.append(value)
            
            _jacobian.append(line)
            
        _jacobian = numpy.array(_jacobian)
                
    return _jacobian


def _build_simpletom_jacobian2(estimate):
    """Try to speed up jacobian building using map"""
    pass   
    
    
        
def _build_simpletom_first_deriv():
    """
    Build the first derivative finite differences matrix for the model space
    """
    
    assert _mesh is not None, "Can't build simpletom derivative matrix." + \
        "No mesh defined"
    
    ny, nx = _mesh.shape
            
    deriv_num = (nx - 1)*ny + (ny - 1)*nx
                
    first_deriv = numpy.zeros((deriv_num, nx*ny))
        
    deriv_i = 0
        
    # Derivatives in the x direction        
    param_i = 0
    for i in xrange(ny):
        
        for j in xrange(nx - 1):                
            
            first_deriv[deriv_i][param_i] = 1
            
            first_deriv[deriv_i][param_i + 1] = -1
            
            deriv_i += 1
            
            param_i += 1
        
        param_i += 1
            
    # Derivatives in the y direction        
    param_i = 0
    for i in xrange(ny - 1):
        
        for j in xrange(nx):
    
            first_deriv[deriv_i][param_i] = 1
            
            first_deriv[deriv_i][param_i + nx] = -1
            
            deriv_i += 1
            
            param_i += 1        
            
    return first_deriv


def _calc_simpletom_adjustment(estimate):
    """Calculate the adjusted data produced by a given estimate"""
    
    global _jacobian
    
    if _jacobian is None:
        
        _jacobian = _build_simpletom_jacobian(estimate)
        
    adjusted = numpy.dot(_jacobian, estimate)
    
    return adjusted


def make_mesh(x1, x2, y1, y2, nx, ny):
    """
    Make a model space discretization mesh.
    
    Parameters:
      
      x1, x2: lower and upper limits of the model space in the x direction
      
      y1, y2: lower and upper limits of the model space in the y direction
      
      nx, ny: number of cells in the x and y directions
      
    Return:
    
      2D array of cells. Each cell is a dictionary as:
        {'x1':cellx1, 'x2':cellx2, 'y1':celly1, 'y2':celly2}
    """
    
    log.info("Building model space mesh:")
    log.info("  Discretization: nx=%d X ny=%d = %d parameters" 
             % (nx, ny, nx*ny))
    
    dx = float(x2 - x1)/nx
    dy = float(y2 - y1)/ny
    
    mesh = []
    
    for i, celly1 in enumerate(numpy.arange(y1, y2, dy)):
        
        # To ensure that there are the right number of cells. arange sometimes
        # makes more cells because of floating point rounding
        if i >= ny:
            
            break
        
        line = []
        
        for j, cellx1 in enumerate(numpy.arange(x1, x2, dx)):
            
            if j >= nx:
                
                break
            
            cell = {'x1':cellx1, 'x2':cellx1 + dx, 
                    'y1':celly1, 'y2':celly1 + dy}
            
            line.append(cell)
            
        mesh.append(line)
            
    return numpy.array(mesh)
        
        
def solve(data, mesh, initial=None, damping=0, smoothness=0, curvature=0, 
          sharpness=0, beta=10**(-5), max_it=100, lm_start=1, lm_step=10, 
          max_steps=20):
    """
    Solve the tomography problem for a given data set and model space mesh.
    
    Parameters:
    
      data: travel time data in a dictionary (as loaded by fatiando.seismo.io)
      
      mesh: model space discretization mesh (see make_mesh function)
      
      initial: initial estimate (only used when given sharpness > 0). If None,
               will use zero initial estimate
      
      damping: Tikhonov order 0 regularization parameter. Must be >= 0
      
      smoothness: Tikhonov order 1 regularization parameter. Must be >= 0
      
      curvature: Tikhonov order 2 regularization parameter. Must be >= 0
      
      sharpness: Total Variation regularization parameter. Must be >= 0
      
      beta: small constant used to make Total Variation differentiable. 
            Must be >= 0. The smaller it is, the sharper the solution but also 
            the less stable
    
      max_it: maximum number of iterations 
        
      lm_start: initial Marquardt parameter (controls the step size)
    
      lm_step: factor by which the Marquardt parameter will be reduced with
               each successful step
             
      max_steps: how many times to try giving a step before exiting
      
    Return:
    
      [estimate, goals]:
        estimate = array-like parameter vector estimated by the inversion
                   parameters are the velocity values in the mesh cells
                   use estimate2matrix function to convert it to the shape of 
                   mesh
        goals = list of goal function value per iteration    
    """
    
    log.info("Inversion parameters:")
    log.info("  damping    = %g" % (damping))
    log.info("  smoothness = %g" % (smoothness))
    log.info("  curvature  = %g" % (curvature))
    log.info("  sharpness  = %g" % (sharpness))
    log.info("  beta       = %g" % (beta))
        
    global _mesh, _sources, _receivers
    
    _mesh = mesh
    
    _sources = data['src']
    
    _receivers = data['rec']
    
    data_vector = data['traveltime']

    solvers.clear()
    
    solvers.damping = damping
    solvers.smoothness = smoothness
    solvers.curvature = curvature
    solvers.sharpness = sharpness
    solvers.beta = beta
    
    solvers._build_jacobian = _build_simpletom_jacobian
    solvers._build_first_deriv_matrix = _build_simpletom_first_deriv
    solvers._calc_adjustment = _calc_simpletom_adjustment
    
    if initial is None:
        
        initial = (10**(-10))*numpy.ones(mesh.size)
        
    else:
        
        initial = 1./numpy.array(initial)
    
    estimate, goals = solvers.lm(data_vector, None, initial, lm_start, lm_step, 
                                 max_steps, max_it)

    # The inversion outputs a slowness estimate. Convert it to velocity
    estimate = 1./numpy.array(estimate)

    return estimate, goals


def residuals(data, estimate):
    """
    Calculate the residuals produced by a given estimate.
    
    Parameters:
    
      data: travel time data in a dictionary (as loaded by fatiando.seismo.io)
    
      estimate: array-like parameter vector produced by the inversion.
      
    Return:
    
      array-like vector of residuals
    """

    adjusted = _calc_simpletom_adjustment(1./estimate)

    residuals = numpy.array(data['traveltime']) - adjusted
    
    return residuals


def fill_mesh(estimate, mesh):
    """
    Fill the 'value' keys of mesh with the values in the estimate
    
    Parameters:
    
      estimate: array-like parameter vector produced by the inversion
      
      mesh: model space discretization mesh used in the inversion to produce the
            estimate (see make_mesh function)
    """
            
    estimate_matrix = numpy.reshape(estimate, mesh.shape)
        
    for i, line in enumerate(mesh):
        
        for j, cell in enumerate(line):
            
            cell['value'] = estimate_matrix[i][j]
            
            
def copy_mesh(mesh):
    """
    Make a copy of mesh.
    Use this instead of numpy.copy or mesh.copy because they don't copy the
    cells.
    """
    
    copy = []
    
    for line in mesh:
        
        copy_line = []
        
        for cell in line:
            
            copy_line.append(cell.copy())
            
        copy.append(copy_line)
        
    return numpy.array(copy)
