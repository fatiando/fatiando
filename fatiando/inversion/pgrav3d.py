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
  * solve: Solve the inverse problem for a given data set and model space mesh
  * clear: Erase garbage from previous inversions.
  * fill_mesh: Fill the 'value' keys of mesh with the inversion estimate.
  * extract_data_vector: Put all the gravity field data in a single array.
  * cal_adjustment: Calculate the adjusted data produced by a given estimate.
  * residuals: Calculate the residuals produced by a given estimate
  * use_depth_weights :Use depth weighting in the next inversions
  * set_bounds: Set lower and upper bounds on the density values
  * grow: Grow the solution around given 'seeds' 
  * get_seed: Returns as a seed the cell in mesh that has point inside it
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
# The distances to the mass elements in the Minimum Moment of Inertia 
# regularization
_distances = None
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
           _depth_weights, _solvers_tk_weights, _distances
               
    _jacobian = None
    _mesh = None
    _data = None
    _data_vector = None
    _depth_weights = None
    _solvers_tk_weights = None
    _distances = None
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
        
    return  numpy.array(data_vector)


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
        
        adjusted_grid = {}
                
        for field in ['gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']:
            
            if field in _data.keys():
                
                adjusted_grid[field] = _data[field].copy()
                
                ndata = len(_data[field]['x'])
                          
                adjusted_grid[field]['value'] = adjusted[:ndata]
                
                adjusted_grid[field]['error'] = None       
                
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


def _calc_mmi_goal(estimate, mmi, power, seeds):
    """Calculate the goal function due to MMI regularization"""
    
    if mmi == 0:
        
        return 0, ''
    
    global _distances
    
    if _distances is None:
        
        _distances = numpy.zeros(_mesh.size)
        
        for i, cell in enumerate(_mesh.ravel()):
                
            dx = float(cell['x2'] - cell['x1'])
            dy = float(cell['y2'] - cell['y1'])
            dz = float(cell['z2'] - cell['z1'])
            
            best_distance = None
            
            for seed in seeds:
                
                x_distance = abs(cell['x1'] - seed['cell']['x1'])/dx
                y_distance = abs(cell['y1'] - seed['cell']['y1'])/dy
                z_distance = abs(cell['z1'] - seed['cell']['z1'])/dz
                
                distance = max([x_distance, y_distance, z_distance])
                
                if best_distance is None or distance < best_distance:
                    
                    best_distance = distance
                     
            _distances[i] = best_distance
    
    weights = (_distances**power)
    
    weights = weights/(weights.max())
    
    goal = mmi*((estimate**2)*weights).sum()
    
    msg = ' MMI:%g' % (goal)
    
    return goal, msg


def get_seed(point, density, mesh):
    """Returns as a seed the cell in mesh that has point inside it."""
    
    x, y, z = point
    
    seed = None
    
    for i, cell in enumerate(mesh.ravel()):
        
        if (x >= cell['x1'] and x <= cell['x2'] and y >= cell['y1'] and  
            y <= cell['y2'] and z >= cell['z1'] and z <= cell['z2']):
            
            seed = {'param':i, 'density':density}
            
            break
        
    if seed is None:
        
        raise ValueError("There is no cell in 'mesh' with 'point' inside it.")
    
    log.info("  seed: %s" % (str(seed)))

    return seed


def _add_neighbors(param, neighbors, mesh, estimate):
    """Add the neighbors of 'param' in 'mesh' to 'neighbors'."""
    
    nz, ny, nx = mesh.shape
    
    append = neighbors.append
    
    # The guy above
    neighbor = param - nx*ny
    above = None
    
    if neighbor > 0:
        
        above = neighbor
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
        
            append(neighbor)
    
    # The guy bellow
    neighbor = param + nx*ny
    bellow = None
    
    if neighbor < mesh.size:
        
        bellow = neighbor
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)    
    
    # The guy in front
    neighbor = param + 1
    front = None
    
    if param%nx < nx - 1:
        
        front = neighbor
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
        
            append(neighbor)
    
    # The guy in the back
    neighbor = param - 1
    back = None
    
    if param%nx != 0:
        
        back = neighbor
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)
    
    # The guy to the left
    neighbor = param + nx
    left = None
    
    if param%(nx*ny) < nx*(ny - 1):
        
        left = neighbor
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
        
            append(neighbor)
    
    # The guy to the right
    neighbor = param - nx
    right = None
    
    if param%(nx*ny) >= nx:
        
        right = neighbor
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)

    # The diagonals            
    if front is not None and left is not None:
        
        neighbor = left + 1
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)
    
    if front is not None and right is not None:
        
        neighbor = right + 1
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)
            
    if back is not None and left is not None:
        
        neighbor = left - 1
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)
            
    if back is not None and right is not None:
    
        neighbor = right - 1
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)
            
    if above is not None and left is not None:
        
        neighbor = above + nx
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)
            
    if above is not None and right is not None:
        
        neighbor = above - nx
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)
            
    if above is not None and front is not None:
        
        neighbor = above + 1
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)
            
    if above is not None and back is not None:
        
        neighbor = above - 1
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)            
            
    if above is not None and front is not None and left is not None:
        
        neighbor = above + nx + 1
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)
            
    if above is not None and front is not None and right is not None:
        
        neighbor = above - nx + 1       
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)

    if above is not None and back is not None and left is not None:
        
        neighbor = above + nx - 1 
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)
    
    if above is not None and back is not None and right is not None:
        
        neighbor = above - nx - 1 
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)
            
    if bellow is not None and left is not None:
        
        neighbor = bellow + nx
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)
            
    if bellow is not None and right is not None:
        
        neighbor = bellow - nx
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)
            
    if bellow is not None and front is not None:
        
        neighbor = bellow + 1
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)
            
    if bellow is not None and back is not None:
        
        neighbor = bellow - 1
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)         
            
    if bellow is not None and front is not None and left is not None:
        
        neighbor = bellow + nx + 1
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)
            
    if bellow is not None and front is not None and right is not None:
        
        neighbor = bellow - nx + 1
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)
        
    if bellow is not None and back is not None and left is not None:
        
        neighbor =  bellow + nx - 1
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)
            
    if bellow is not None and back is not None and right is not None:
        
        neighbor = bellow - nx - 1
        
        if neighbor not in neighbors and estimate[neighbor] == 0.:
            
            append(neighbor)
        

def grow(data, mesh, seeds, mmi, power=5, apriori_variance=1):
    """
    Grow the solution around given 'seeds'.
    
    Parameters:
        
      data: dictionary with the gravity component data as:
            {'gz':gzdata, 'gxx':gxxdata, 'gxy':gxydata, ...}
            If there is no data for a given component, omit the respective key.
            Each g*data is a data grid as loaded by fatiando.gravity.io
      
      mesh: model space discretization mesh (see geometry.prism_mesh function)
      
      seeds: list of seeds (to make a seed, see get_seed function)
      
      mmi: Minimum Moment of Inertia regularization parameter (how compact the
           solution should be around the seeds). Has to be >= 0
           
      power: power to which the distances are raised in the MMI weights
           
      apriori_variance: a priori variance of the data
      
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
    log.info("  mmi = %g" % (mmi))
    
    global _mesh, _data, _jacobian

    _mesh = mesh
    _data = data
    
    _build_pgrav3d_jacobian(None)
    
    _jacobian = _jacobian.T
        
    estimate = numpy.zeros(_mesh.size)
    
    # Need to set their densities before so that seeds won't be added as 
    # neighbors
    for seed in seeds:
        
        estimate[seed['param']] = seed['density']
    
    for seed in seeds:
                
        seed['neighbors'] = []
        
        _add_neighbors(seed['param'], seed['neighbors'], mesh, estimate)
                
        seed['cell'] = _mesh.ravel()[seed['param']]
        
    reg_goal, msg = _calc_mmi_goal(estimate, mmi, power, seeds)
    
    residuals = extract_data_vector(data) - numpy.dot(_jacobian.T, estimate)
    
    rms = (residuals*residuals).sum()
    
    goals = [rms + reg_goal] 
    
    log.info("Growing density model:")
    log.info("  parameters = %d" % (mesh.size))
    log.info("  data = %d" % (len(residuals)))
    log.info("  initial RMS = %g" % (rms))
    log.info("  initial regularizer goals =%s" % (msg))
    log.info("  initial total goal function = %g" % (goals[-1]))
    
    total_start = time.time()
    
    marked = []
        
    for iteration in xrange(mesh.size - len(seeds)):
        
        start = time.time()
        
        log.info("  it %d:" % (iteration + 1))
        
        stagnation = True
            
        # Want to find the neighbor (of all seeds) that best reduces the goal 
        # function
        best_param = None
        best_goal = goals[-1]
        best_rms = None
        best_msg = ''
        best_seed = None
        
        for seed_num, seed in enumerate(seeds):
            
            density = seed['density']
            
            for neighbor in seed['neighbors']:
                
                new_residuals = residuals - density*_jacobian[neighbor]
                
                rms = (new_residuals*new_residuals).sum()
                
                estimate[neighbor] = density
    
                reg_goal, msg = _calc_mmi_goal(estimate, mmi, power, seeds)
                
                estimate[neighbor] = 0
                
                goal = rms + reg_goal
                
                if goal < best_goal:
                    
                    best_param = neighbor
                    best_goal = goal
                    best_rms = rms
                    best_msg = msg
                    best_density = density
                    best_seed = seed_num
                    stagnation = False
                 
        if best_param is not None:
            
            estimate[best_param] = best_density
            residuals -= best_density*_jacobian[best_param]
            goals.append(best_goal)
            marked.append([best_param, best_seed])
            
            # Remove the best_param from all lists of neighbors
            for seed in seeds:
                
                try:          
                                                          
                    seed['neighbors'].remove(best_param)      
                                             
                except ValueError:
                    
                    pass
                
            _add_neighbors(best_param, seeds[best_seed]['neighbors'], mesh, 
                            estimate)
            
            log.info("    append to seed %d: RMS=%g%s TOTAL=%g" 
                     % (best_seed + 1, best_rms, best_msg, best_goal))
                                    
        if stagnation:
                    
            # If couldn't add any prism to the solution, try moving the ones
            # already marked around.
            # NOTE: The way it is, the cells are only remanaged within the same
            # seed, not from one seed to another
            stagnation_again = True
                                            
            best_goal = goals[-1]
            best_param = None
            param_to_remove = None
            best_rms = None
            best_msg = ''
            best_seed = None
            best_neighbors = None
            best_density = None
            
            for param, seed_id in reversed(marked):
                
                # Remove the param from the estimate and residuals
                density = estimate[param]                
                estimate[param] = 0
                tmp_residuals = residuals + density*_jacobian[param]
                
                # Re-calculate the neighbors so that the param is included
                # in the list and it's sole neighbors are excluded.
                neighbors = []
                
                for other_param, other_seed in marked:
                    
                    if other_param != param and other_seed == seed_id:
                        
                        # Only the ones with estimate == 0
                        _add_neighbors(other_param, neighbors, mesh, estimate)
                                                   
                for neighbor in neighbors:
                                        
                    new_residuals = tmp_residuals - density*_jacobian[neighbor]
                    
                    rms = (new_residuals*new_residuals).sum()
    
                    estimate[neighbor] = density
                    
                    reg_goal, msg = _calc_mmi_goal(estimate, mmi, power, seeds)
                    
                    estimate[neighbor] = 0
                    
                    goal = rms + reg_goal
                    
                    if goal < best_goal:
                        
                        best_param = neighbor
                        best_goal = goal
                        best_rms = rms
                        best_msg = msg
                        best_seed = seed_id
                        best_neighbors = neighbors
                        best_density = density
                        param_to_remove = param
                        
                        stagnation_again = False
                        
                # Return things to normal
                estimate[param] = density
                    
            if not stagnation_again:
                          
                # Remove the parameter that is to be remanaged
                estimate[param_to_remove] = 0
                residuals += best_density*_jacobian[param_to_remove]
                
                # Put the remanaged parameter in the estimate
                estimate[best_param] = best_density
                goals.append(best_goal)
                residuals -= best_density*_jacobian[best_param]
                
                # Change the neighbors in the seed that was remanaged
                seeds[best_seed]['neighbors'] = best_neighbors
                
                # Remove the best_param from all lists of neighbors
                for seed in seeds:
                                  
                    try:         
                                                               
                        seed['neighbors'].remove(best_param)    
                                                        
                    except ValueError:
                        
                        pass
                    
                _add_neighbors(best_param, seeds[best_seed]['neighbors'], 
                                mesh, estimate)
                                
                log.info("    remanaged in seed %d: RMS=%.20g%s TOTAL=%g" 
                         % (best_seed + 1, best_rms, best_msg, best_goal))
                
            else:
                                
                log.warning("    Exited because couldn't grow or remanage.")
                break
                
        aposteriori_variance = goals[-1]/float(len(residuals))
        
        log.info("    a posteriori variance = %g" % (aposteriori_variance))                    
                    
        end = time.time()
        log.info("    time: %g s" % (end - start))
        
        if aposteriori_variance <= 1.1*apriori_variance and \
           aposteriori_variance >= 0.9*apriori_variance:
            
            break 
    
    _jacobian = _jacobian.T
    
    total_end = time.time()
    
    log.info("  Total inversion time: %g s" % (total_end - total_start))

    return estimate, goals