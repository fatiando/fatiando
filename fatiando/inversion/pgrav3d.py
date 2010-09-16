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
3D Gravity inversion using right rectangular prisms.

Functions:
  * clear: Erase garbage from previous inversions.
  * fill_mesh: Fill the 'value' keys of mesh with the inversion estimate.
  * extract_data_vector: Put all the gravity field data in a single array.
  * cal_adjustment: Calculate the adjusted data produced by a given estimate.
  * residuals: Calculate the residuals produced by a given estimate
  * use_depth_weights :Use depth weighting in the next inversions
  * set_bounds: Set lower and upper bounds on the density values
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 14-Jun-2010'

import time
import logging

import numpy

import fatiando
import fatiando.gravity.prism
from fatiando.inversion import solvers
        
log = logging.getLogger('fatiando.inversion.pgrav3d')       
log.setLevel(logging.DEBUG)
log.addHandler(fatiando.default_log_handler)


# The Jacobian only needs to be calculate once per data and mesh
_jacobian = None
# Depth weights are also only calculated once
_depth_weights = None
# This will hold solvers._build_tk_weights so that I don't loose when it's
# overwritten with _build_tk_depth_weights
_solvers_tk_weights = None
# Keep the mesh and data global to access them to build the Jacobian and derivs
_mesh = None
_data = None
_data_vector = None
_calculators = {'gz':fatiando.gravity.prism.gz,
                'gxx':fatiando.gravity.prism.gxx,
                'gxy':fatiando.gravity.prism.gxy,
                'gxz':fatiando.gravity.prism.gxz,
                'gyy':fatiando.gravity.prism.gyy,
                'gyz':fatiando.gravity.prism.gyz,
                'gzz':fatiando.gravity.prism.gzz}


def clear():
    """
    Erase garbage from previous inversions.
    Only use if changing the data and/or mesh (otherwise it saves time to keep
    the garbage)
    """
    
    global _jacobian, _mesh, _data, _data_vector, \
           _depth_weights, _solvers_tk_weights
               
    _jacobian = None
    _mesh = None
    _data = None
    _data_vector = None
    _depth_weights = None
    _solvers_tk_weights = None
    reload(solvers)
    

def fill_mesh(estimate, mesh):
    """
    Fill the 'value' keys of mesh with the values in the inversion estimate
    
    Parameters:
    
      estimate: array-like parameter vector produced by the inversion
      
      mesh: model space discretization mesh used in the inversion to produce the
            estimate (see fatiando.geometry.prism_mesh function)
    """
    
    for value, cell in zip(estimate, mesh.ravel()):
        
        cell['value'] = value


def extract_data_vector(data, inplace=False):
    """
    Put all the gravity field data in a single array for use in inversion.
    
    Parameters:
    
      data: dictionary with the gravity component data as:
            {'gz':gzdata, 'gxx':gxxdata, 'gxy':gxydata, ...}
            If there is no data for a given component, omit the respective key.
            Each g*data is a data grid as loaded by fatiando.gravity.io
            
      inplace: wether or not to erase the values in 'data' as they are put into
               the array (use to save memory when data set is large)
    
    Return:
        
      1D array-like with the data in the following order:
        gz, gxx, gxy, gxz, gyy, gyz, gzz
    """
    
    data_vector = []
    
    if 'gz' in data.keys():
        
        data_vector.extend(data['gz']['value'])
        
        if inplace:
            
            del data['gz']['value']
        
    if 'gxx' in data.keys():
        
        data_vector.extend(data['gxx']['value'])
        
        if inplace:
            
            del data['gxx']['value']
        
    if 'gxy' in data.keys():
        
        data_vector.extend(data['gxy']['value'])
        
        if inplace:
            
            del data['gxy']['value']
        
    if 'gxz' in data.keys():
        
        data_vector.extend(data['gxz']['value'])
        
        if inplace:
            
            del data['gxz']['value']
        
    if 'gyy' in data.keys():
        
        data_vector.extend(data['gyy']['value'])
        
        if inplace:
            
            del data['gyy']['value']
        
    if 'gyz' in data.keys():
        
        data_vector.extend(data['gyz']['value'])
        
        if inplace:
            
            del data['gyz']['value']
        
    if 'gzz' in data.keys():
        
        data_vector.extend(data['gzz']['value'])
        
        if inplace:
            
            del data['gzz']['value']
        
    return  data_vector


def residuals(data, estimate):
    """
    Calculate the residuals produced by a given estimate.
    
    Parameters:
    
      data: gravity field data in a dictionary (as loaded by 
            fatiando.gravity.io)
    
      estimate: array-like parameter vector produced by the inversion.
      
    Return:
    
      array-like vector of residuals
    """       

    adjusted = calc_adjustment(estimate)
    
    key = data.keys()[0]
    
    if 'value' in data[key].keys():
        
        data_vector = extract_data_vector(data)
        
    else:
        
        assert _data_vector is not None, \
            "Missing 'value' key in %s data" % (key)
            
        data_vector = _data_vector        

    residuals = data_vector - adjusted
    
    return residuals


def calc_adjustment(estimate, grid=False):
    """
    Calculate the adjusted data produced by a given estimate.
    
    Parameters:
    
      estimate: array-like parameter vector produced by the inversion.
      
      grid: if True, return a dictionary of grids like the one given to solve
            function.
            (grids compatible with load and dump in fatiando.gravity.io and the
             plotters in fatiando.visualization).
            if False, return a data vector to use in inversion
    """
    
    jacobian = _build_pgrav3d_jacobian(estimate)
    
    adjusted = numpy.dot(jacobian, estimate)
    
    if grid:
        
        adjusted_grid = _data.copy()
                
        for field in ['gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']:
            
            if field in _data.keys():
                
                ndata = len(_data[field]['x'])
                          
                adjusted_grid[field]['value'] = adjusted[:ndata]
                
                adjusted = adjusted[ndata:]                
    
        adjusted = adjusted_grid
        
    return adjusted


def _build_pgrav3d_jacobian(estimate):
    """Build the Jacobian matrix of the gravity field"""
    
    assert _mesh is not None, "Can't build Jacobian. No mesh defined"
    assert _data is not None, "Can't build Jacobian. No data defined"
    
    global _jacobian
    
    if _jacobian is None:
        
        start = time.time()
        
        _jacobian = []
        append_row = _jacobian.append
        
        for field in ['gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']:
            
            if field in _data.keys():
                
                coordinates =  zip(_data[field]['x'], _data[field]['y'], 
                                   _data[field]['z'])
                
                function = _calculators[field]
                
                for x, y, z in coordinates:
                    
                    row = [function(1., cell['x1'], cell['x2'], cell['y1'], 
                                    cell['y2'], cell['z1'], cell['z2'], 
                                    x, y, z)
                           for cell in _mesh.ravel()]
                        
                    append_row(row)
                    
        _jacobian = numpy.array(_jacobian)
        
        end = time.time()
        
        log.info("  Built Jacobian (sensibility) matrix (%g s)"
                 % (end - start))
        
    return _jacobian                   


def _build_pgrav3d_first_deriv():
    """
    Build the first derivative finite differences matrix for the model space
    """
    
    assert _mesh is not None, "Can't build first derivative matrix." + \
        "No mesh defined"
        
    nz, ny, nx = _mesh.shape
                
    deriv_num = (nx - 1)*ny*nz + (ny - 1)*nx*nz + (nz - 1)*nx*ny
            
    first_deriv = numpy.zeros((deriv_num, nx*ny*nz))
    
    deriv_i = 0
    
    # Derivatives in the x direction        
    param_i = 0
    
    for k in xrange(nz):
        
        for j in xrange(ny):
            
            for i in xrange(nx - 1):                
                
                first_deriv[deriv_i][param_i] = 1
                
                first_deriv[deriv_i][param_i + 1] = -1
                
                deriv_i += 1
                
                param_i += 1
            
            param_i += 1
        
    # Derivatives in the y direction        
    param_i = 0
    
    for k in xrange(nz):
    
        for j in range(ny - 1):
            
            for i in range(nx):
        
                first_deriv[deriv_i][param_i] = 1
                
                first_deriv[deriv_i][param_i + nx] = -1
                
                deriv_i += 1
                
                param_i += 1
                
        param_i += nx
        
    # Derivatives in the z direction        
    param_i = 0
    
    for k in xrange(nz - 1):
    
        for j in range(ny):
            
            for i in range(nx):
        
                first_deriv[deriv_i][param_i] = 1
                
                first_deriv[deriv_i][param_i + nx*ny] = -1
                
                deriv_i += 1
                
                param_i += 1
    
    return first_deriv


def _build_tk_depth_weights(nparams):
    """
    Build the Tikhonov weights using depth weighting (Li & Oldenburg, 1998).
    """
        
    weights = _solvers_tk_weights(nparams)
    
    for i, row in enumerate(weights):
        
        row *= _depth_weights[i]*_depth_weights
        
    return weights


def _calc_adjusted_depth_weights(coefs):
    """Calculate the adjusted depth weights for a given set of coefficients"""
    
    assert _mesh is not None, \
        "Can't calculate adjusted depth weights without a mesh"
    
    z0, power = coefs
    
    weights = numpy.zeros(_mesh.size)
    
    for i, cell in enumerate(_mesh.ravel()):
        
        depth = 0.5*(cell['z1'] + cell['z2'])
        
        weights[i] = (depth + z0)**(-0.5*power)
        
    return weights


def _build_depth_weights_jacobian(estimate):
    """Build the Jacobian of the depth weighing function"""
    
    jacobian = []
    
    z0, power = estimate
                    
    for cell in _mesh.ravel():
        
        depth = 0.5*(cell['z1'] + cell['z2'])
                    
        z0_deriv = -0.5*power*(z0 + depth)**(-0.5*(power + 2))
        
        power_deriv = -0.5*(z0 + depth)**(-0.5*power)
        
        jacobian.append([z0_deriv, power_deriv])
        
    return numpy.array(jacobian)    
    

def use_depth_weights(mesh, z0=None, power=None, grid_height=None, 
                      normalize=True):
    """
    Use depth weighting in the next inversions (Li & Oldenburg, 1998).
    
    If z0 or power are set to None, they will be automatically calculated.
    
    Parameters:
    
      mesh: model space discretization mesh (see geometry.prism_mesh function)
    
      z0: compensation depth
      
      power: power of the power law used
      
      grid_height: height of the data grid in meters (only needed is z0 and 
                   power are None)
      
      normalize: whether or not to normalize the weights
    """
    
    if z0 is None or power is None:
    
        log.info("Adjusting depth weighing coefficients:")
        
        import fatiando.inversion.solvers as local_solver
        
        global _mesh
        
        _mesh = mesh
        
        # Make a 'data' array (gzz kernel values)
        kernel_data = []
        
        for cell in mesh.ravel():
            
            x = 0.5*(cell['x1'] + cell['x2'])
            y = 0.5*(cell['y1'] + cell['y2'])
            
            kernel = fatiando.gravity.prism.gzz(1., cell['x1'], cell['x2'], 
                                                cell['y1'], cell['y2'], 
                                                cell['z1'], cell['z2'], 
                                                x, y, -grid_height)
            
            kernel_data.append(kernel)
            
        local_solver._build_jacobian = _build_depth_weights_jacobian
        local_solver._calc_adjustment = _calc_adjusted_depth_weights
        
        coefs, goals = local_solver.lm(kernel_data, None, 
                                       numpy.array([1., 3.]))
        
        z0, power = coefs
        
        _mesh = None
    
    log.info("Setting depth weighting:")
    log.info("  z0 = %g" % (z0))
    log.info("  power = %g" % (power))
    log.info("  normalized = %s" % (str(normalize)))
                  
    weights = numpy.zeros(mesh.size)
    
    for i, cell in enumerate(mesh.ravel()):
        
        depth = 0.5*(cell['z1'] + cell['z2'])
        
        weights[i] = (depth + z0)**(-0.25*power)
        
    if normalize:
        
        weights = weights/weights.max()
        
    global _depth_weights
    
    _depth_weights = weights
        
    # Overwrite the default Tikhonov weights builder but not before saing it
    global _solvers_tk_weights
    
    _solvers_tk_weights = solvers._build_tk_weights
    
    solvers._build_tk_weights = _build_tk_depth_weights

    return z0, power


def set_bounds(lower, upper):
    """Set lower and upper bounds on the density values"""
    
    solvers.set_bounds(lower, upper)


def solve(data, mesh, initial=None, damping=0, smoothness=0, curvature=0, 
          sharpness=0, beta=10**(-5), compactness=0, epsilon=10**(-5), 
          max_it=100, lm_start=1, lm_step=10, max_steps=20):    
    """
    Solve the inverse problem for a given data set and model space mesh.
    
    Parameters:
    
      data: dictionary with the gravity component data as:
            {'gz':gzdata, 'gxx':gxxdata, 'gxy':gxydata, ...}
            If there is no data for a given component, omit the respective key.
            Each g*data is a data grid as loaded by fatiando.gravity.io
      
      mesh: model space discretization mesh (see geometry.prism_mesh function)
      
      initial: initial estimate (only used with sharpness or compactness). 
               If None, will use zero initial estimate
      
      damping: Tikhonov order 0 regularization parameter. Must be >= 0
      
      smoothness: Tikhonov order 1 regularization parameter. Must be >= 0
      
      curvature: Tikhonov order 2 regularization parameter. Must be >= 0
      
      sharpness: Total Variation regularization parameter. Must be >= 0
      
      beta: small constant used to make Total Variation differentiable. 
            Must be >= 0. The smaller it is, the sharper the solution but also 
            the less stable
            
      compactness: Compact regularization parameter. Must be >= 0
      
      epsilon: small constant used in Compact regularization to avoid 
               singularities. Set it small for more compactness, larger for more
               stability.
    
      max_it: maximum number of iterations 
        
      lm_start: initial Marquardt parameter (controls the step size)
    
      lm_step: factor by which the Marquardt parameter will be reduced with
               each successful step
             
      max_steps: how many times to try giving a step before exiting
      
    Return:
    
      [estimate, goals]:
        estimate = array-like parameter vector estimated by the inversion.
                   parameters are the density values in the mesh cells.
                   use fill_mesh function to put the estimate in a mesh so you
                   can plot and save it.
        goals = list of goal function value per iteration    
    """

    for key in data.keys():
        assert key in ['gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz'], \
            "Invalid gravity component (data key): %s" % (key)
    
    log.info("Inversion parameters:")
    log.info("  damping     = %g" % (damping))
    log.info("  smoothness  = %g" % (smoothness))
    log.info("  curvature   = %g" % (curvature))
    log.info("  sharpness   = %g" % (sharpness))
    log.info("  beta        = %g" % (beta))
    log.info("  compactness = %g" % (compactness))
    log.info("  epsilon     = %g" % (epsilon))
    
    global _mesh, _data, _data_vector

    _mesh = mesh
    _data = data
        
    _data_vector = extract_data_vector(data)
    
    log.info("  parameters = %d" % (mesh.size))
    log.info("  data = %d" % (len(_data_vector)))

    if initial is None:
        
        initial = 10**(-7)*numpy.ones(mesh.size)
                        
    # Overwrite the needed methods for solvers to work
    solvers._build_jacobian = _build_pgrav3d_jacobian
    solvers._build_first_deriv_matrix = _build_pgrav3d_first_deriv
    solvers._calc_adjustment = calc_adjustment
    
    solvers.damping = damping
    solvers.smoothness = smoothness
    solvers.curvature = curvature
    solvers.sharpness = sharpness
    solvers.beta = beta
    solvers.compactness = compactness
    solvers.epsilon = epsilon
    
    estimate, goals = solvers.lm(_data_vector, None, initial, lm_start, lm_step, 
                                 max_steps, max_it)

    return estimate, goals




#    def _build_jacobian(self, estimate):
#        """
#        Make the Jacobian matrix of the function of the parameters.
#        """
#        
#        assert estimate != None, "Can't use solve_linear. " + \
#            "This is a non-linear inversion!"
#        
#        jacobian = []
#        
#        z0 = estimate[0]
#        
#        power = estimate[1]
#                        
#        for z in self._depths:
#                        
#            z0_deriv = -power/((z + 0.5*self._dz + z0)**(power + 1))
#            
#            power_deriv = -1./((z + 0.5*self._dz + z0)**power)
#            
#            jacobian.append([z0_deriv, power_deriv])
#            
#        return numpy.array(jacobian)        
#            
#    
#    def _calc_adjusted_data(self, estimate):
#        """
#        Calculate the adjusted data vector based on the current estimate
#        """
#        
#        z0 = estimate[0]
#        
#        power = estimate[1]
#        
#        adjusted = []
#        
#        for z in self._depths: 
#        
#            adjusted.append(1./((z + 0.5*self._dz + z0)**power))
#            
#        return numpy.array(adjusted)

#    def _get_data_array(self):
#        """
#        Return the data in a Numpy array so that the algorithm can access it
#        in a general way
#        """        
#        
#        if self._data != None:
#            
#            return self._data
#        
#        data = []
#        
#        dx = (self._pgrav_solver._mod_x2 - self._pgrav_solver._mod_x1)/ \
#             self._pgrav_solver._nx
#             
#        dy = (self._pgrav_solver._mod_y2 - self._pgrav_solver._mod_y1)/ \
#             self._pgrav_solver._ny
#                        
#        for depth in self._depths:
#            
#            tmp = prism_gravity.gzz(1., -0.5*dx, 0.5*dx, -0.5*dy, 0.5*dy, \
#                            depth, depth + self._dz, 0., 0., -self._height)
#                
#            data.append(tmp)
#        
#        self._data = numpy.array(data)
#        
#        return self._data
#    def set_equality(self, z0=None, power=None):
#        """
#        Set an equality constraint for the parameters z0 and/or power.
#        """
#        
#        self._equality_matrix = []
#          
#        self._equality_values = []
#        
#        if z0 != None:
#            
#            self._equality_values.append(z0)
#            
#            self._equality_matrix.append([1, 0])
#            
#        if power != None:
#                        
#            self._equality_values.append(power)
#            
#            self._equality_matrix.append([0, 1])
#            
#        self._equality_values = numpy.array(self._equality_values)
#        
#        self._equality_matrix = numpy.array(self._equality_matrix)
