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
2D gravity inversion of the relief of an interface using rectangular prisms.

Functions:
  * clear: Erase garbage from previous inversions
  * fill_mesh: Fill the 'key' value of each cell of mesh with the values in the 
               estimate
  * adjustment: Calculate the adjusted data produced by a given estimate
  * residuals: Calculate the residuals vector of a given estimate
  * set_bounds: Set bounds on the parameter values (depth of interface)
  * solve: Solve the inversion problem for a given data set and model space mesh
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 05-Jul-2010'

import logging

import numpy

import fatiando
import fatiando.grav.prism
from fatiando.inversion import solvers
import fatiando.inversion.pgrav3d


log = logging.getLogger('fatiando.inversion.interg2d')
log.setLevel(logging.DEBUG)
log.addHandler(fatiando.default_log_handler)


# Keep the mesh and data global to access them to build the Jacobian and derivs
_mesh = None
_data = None
_data_vector = None
_calculators = {'gz':fatiando.grav.prism.gz,
                'gxx':fatiando.grav.prism.gxx,
                'gxy':fatiando.grav.prism.gxy,
                'gxz':fatiando.grav.prism.gxz,
                'gyy':fatiando.grav.prism.gyy,
                'gyz':fatiando.grav.prism.gyz,
                'gzz':fatiando.grav.prism.gzz}

# The reference surface (top of the prisms) and density
_ref_surf = None
_ref_dens = None

# How much to exaggerate the y dimension of the prisms to make them 2D
exaggerate = 1000.
_ysize = None

# The size of the step in the finite differences derivative in the Jacobian
deltaz = 0.5


def clear():
    """
    Erase garbage from previous inversions.
    """
    
    global _mesh, _data, _data_vector, _ref_surf, _ref_dens, _ysize
               
    _mesh = None
    _data = None
    _data_vector = None
    _ref_surf = None
    _ref_dens = None
    _ysize = None
    reload(solvers)


def fill_mesh(estimate, mesh, key='value'):
    """
    Fill the 'key' value of each cell of mesh with the values in the estimate
    
    Parameters:
    
      estimate: array-like parameter vector produced by the inversion
      
      mesh: model space discretization mesh used in the inversion to produce the
            estimate (see geometry.line_mesh function)
            
      key: key in the mesh to fill
    """
        
    for value, cell in zip(estimate, mesh):
        
        cell[key] = value


def _build_interg2d_jacobian(estimate):
    """Build the Jacobian matrix"""

    global _mesh, _data, _ref_surf, _ref_dens, _calculators, _ysize, deltaz

    assert _mesh is not None, "Can't build Jacobian. No mesh defined"
    assert _data is not None, "Can't build Jacobian. No data defined"
    assert _ref_surf is not None, "Can't build Jacobian. " + \
        "No reference surface defined"
    assert _ref_dens is not None, "Can't build Jacobian. " + \
        "No reference density defined"

    jacobian = []
    append_row = jacobian.append
    
    for field in ['gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']:
        
        if field in _data:
            
            function = _calculators[field]
            
            for x, z in zip(_data[field]['x'], _data[field]['z']):
                
                row = []
                append_col = row.append                             
                
                for z1, z2, cell in zip(_ref_surf, estimate, _mesh):
                                              
                    grav1 = function(_ref_dens, cell['x1'], cell['x2'], -_ysize, 
                                     _ysize, z1, z2 - 0.5*deltaz, x, 0., z)
                    
                    grav2 = function(_ref_dens, cell['x1'], cell['x2'], -_ysize, 
                                     _ysize, z1, z2 + 0.5*deltaz, x, 0., z)
                       
                    append_col(float(grav2 - grav1)/deltaz)
                    
                append_row(row)
                
    jacobian = numpy.array(jacobian)
    
    return jacobian


def _build_interg2d_first_deriv():
    """Build the finite differences derivative matrix of the parameters"""
    
    global _mesh
    
    assert _mesh is not None, "Can't build derivative matrix. No mesh defined"
    
    nparams = _mesh.size
    nderivs = (nparams - 1)
    
    first_deriv = numpy.zeros((nderivs, nparams), dtype='f')
            
    # Derivatives in the x direction   
    for i in xrange(nparams - 1):                
        
        first_deriv[i][i] = -1
        
        first_deriv[i][i + 1] = 1
            
    return first_deriv


def adjustment(estimate, profile=False):
    """
    Calculate the adjusted data produced by a given estimate.
    
    Parameters:
    
        estimate: array-like inversion estimate
        
        profile: if True, return the adjusted data in a profile data structure
                 like the one provided to the inversion
    """

    global _data, _ref_surf, _ref_dens, _ysize, _calculators, _mesh, \
           _data_vector
    
    assert _mesh is not None, "Can't calculate adjustment. No mesh defined"
    assert _data is not None, "Can't calculate adjustment. No data defined"
    assert _ref_surf is not None, "Can't calculate adjustment. " + \
        "No reference surface defined"
    assert _ref_dens is not None, "Can't calculate adjustment. " + \
        "No reference density defined"
    
    if not profile:
        
        adjusted = numpy.zeros(len(_data_vector))
        
        for field in ['gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']:
            
            if field in _data.keys():
                
                function = _calculators[field]
                
                for i, coordinates in enumerate(zip(_data[field]['x'], 
                                                    _data[field]['z'])):
                    
                    x, z = coordinates
                                        
                    for z1, z2, cell in zip(_ref_surf, estimate, _mesh):
                        
                        adjusted[i] += function(_ref_dens, 
                                                cell['x1'], cell['x2'], 
                                                -_ysize, _ysize, z1, z2, 
                                                x, 0., z)
                            
    else:
        
        adjusted = {}
        
        for field in ['gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']:
            
            if field in _data.keys():
                
                function = _calculators[field]
                
                adjusted[field] = _data[field].copy()
        
                adjusted[field]['value'] = []
                
                for x, z in zip(adjusted[field]['x'], adjusted[field]['z']):
                    
                    value = 0.
                    
                    for z1, z2, cell in zip(_ref_surf, estimate, _mesh):
                        
                        value += function(_ref_dens, cell['x1'], cell['x2'], 
                                          -_ysize, _ysize, z1, z2, x, 0., z)
        
                    adjusted[field]['value'].append(value)
                
    return adjusted


def residuals(estimate):
    """
    Calculate the residuals vector of a given estimate.
    
    Parameters:
    
      estimate: array-like vector of estimates
      
    Return:
    
      array-like vector of residuals
    """
    
    global _data_vector
    
    assert _data_vector is not None, "Can't calculate residuals. No data vector"
    
    adjusted = adjustment(estimate, profile=False)
    
    residuals = _data_vector - adjusted
    
    return residuals


def set_bounds(vmin, vmax):
    """Set bounds on the parameter values (depth of interface)"""
    
    solvers.set_bounds(vmin, vmax)

        
def solve(data, mesh, density, ref_surf=None, initial=None, damping=0, 
          smoothness=0, curvature=0, sharpness=0, beta=10**(-5), max_it=100, 
          lm_start=1, lm_step=10, max_steps=20):
    """
    Solve the inversion problem for a given data set and model space mesh.
        
    Parameters:    
    
      data: dictionary with the gravity component data as:
            {'gz':gzdata, 'gxx':gxxdata, 'gxy':gxydata, ...}
            If there is no data for a given component, omit the respective key.
            Each g*data is a data profile as loaded by fatiando.grav.io
      
      mesh: model space discretization mesh (see geometry.line_mesh function)
      
      density: density contrast of the basement or interface
      
      ref_surf: a reference surface to be used as the top of the prisms
      
      initial: initial estimate. If None, will use zero initial estimate
      
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
        estimate = array-like parameter vector estimated by the inversion.
                   parameters are the depths of the bottom of the prisms in the 
                   mesh cells. Use fill_mesh function to put the estimate in a 
                   mesh so you can plot and save it.
        goals = list of goal function value per iteration
    """
    
    log.info("Inversion parameters:")
    log.info("  damping    = %g" % (damping))
    log.info("  smoothness = %g" % (smoothness))
    log.info("  curvature  = %g" % (curvature))
    log.info("  sharpness  = %g" % (sharpness))
    log.info("  beta       = %g" % (beta))
        
    global _mesh, _data, _ref_dens, _ref_surf, _data_vector, _ysize, exaggerate
    
    _ref_dens = density
    
    if ref_surf is None:
        
        _ref_surf = numpy.zeros(mesh.size)
        
    else:
        
        _ref_surf = ref_surf 
    
    _mesh = mesh
    
    # Find the biggest cell in mesh and use it to estimate the needed size in y
    biggest = 0
    
    for cell in mesh:
        
        size = abs(cell['x2'] - cell['x1'])
        
        if size > biggest:
            
            biggest = size
            
    _ysize = exaggerate*biggest

    _data = data
    
    _data_vector = fatiando.inversion.pgrav3d.extract_data_vector(data)

    solvers.clear()
    
    solvers.damping = damping
    solvers.smoothness = smoothness
    solvers.curvature = curvature
    solvers.sharpness = sharpness
    solvers.beta = beta
    
    global _build_interg2d_jacobian, _build_interg2d_first_deriv, adjustment
    
    solvers._build_jacobian = _build_interg2d_jacobian
    solvers._build_first_deriv_matrix = _build_interg2d_first_deriv
    solvers._calc_adjustment = adjustment
    
    if initial is None:
        
        initial = (10**(-10))*numpy.ones(mesh.size)
            
    estimate, goals = solvers.lm(_data_vector, None, initial, lm_start, lm_step, 
                                 max_steps, max_it)

    return estimate, goals