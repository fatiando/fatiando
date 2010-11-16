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

* :func:`fatiando.inversion.pgrav3d.clear`
    Erase garbage from previous inversions.

* :func:`fatiando.inversion.pgrav3d.extract_data_vector`
    Put all the gravity field data in a single array.

* :func:`fatiando.inversion.pgrav3d.use_depth_weights`
    Use depth weighting in the next inversions

* :func:`fatiando.inversion.pgrav3d.set_bounds`
    Set lower and upper bounds on the density values
    
* :func:`fatiando.inversion.pgrav3d.solve`    
    Solve the inverse problem for the density using a given data set.
    
* :func:`fatiando.inversion.pgrav3d.adjustment`
    Calculate the adjusted data based on the residuals and the original data.
    
* :func:`fatiando.inversion.pgrav3d.get_seed`
    Returns as a seed the cell in the mesh that has *point* inside it.

* :func:`fatiando.inversion.pgrav3d.grow`
    Invert by growing the solution around given seeds.
    
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 14-Jun-2010'


import time
import logging
import math
import os
import zipfile
import tarfile
import StringIO

import numpy
import pylab

import fatiando
import fatiando.grav.prism
import fatiando.inversion.solvers as solvers
        
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
# The list of supported fields and the calculator function for each one
supported_fileds = ['gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
_calculators = {'gz':fatiando.grav.prism.gz,
                'gxx':fatiando.grav.prism.gxx,
                'gxy':fatiando.grav.prism.gxy,
                'gxz':fatiando.grav.prism.gxz,
                'gyy':fatiando.grav.prism.gyy,
                'gyz':fatiando.grav.prism.gyz,
                'gzz':fatiando.grav.prism.gzz}


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
    

def extract_data_vector(data, inplace=False):
    """
    Put all the gravity field data in a single array for use in inversion.
    
    Parameters:
    
    * data 
        Dictionary with the gravity component data as 
        ``{'gz':gzdata, 'gxx':gxxdata, 'gxy':gxydata, ...}``
        If there is no data for a given component, omit the respective key.
        Each g*data is a profile data dictionary (see bellow)
            
    * inplace
        If ``True`` will erase the values in *data* as they are put into
        the array (use to save memory when data set is large)
    
    Returns:
        
    * data_vector
        1D array-like with the data in the following order:
        gz, gxx, gxy, gxz, gyy, gyz, gzz
        
    The data dictionaries should be as::
    
        {'x':[x1, x2, ...], 'y':[y1, y2, ...], 'z':[z1, z2, ...],
         'value':[data1, data2, ...], 'error':[error1, error2, ...]}
         
    """
    
    global supported_fileds
    
    data_vector = []
    
    for field in supported_fileds:
        
        if field in data:
            
            data_vector.extend(data[field]['value'])
            
            if inplace:
                
                del data[field]['value']
        
    return  numpy.array(data_vector)


def adjustment(data, residuals):
    """
    Calculate the adjusted data based on the residuals and the original data.
    
    Parameters:
    
    * data 
        Dictionary with the gravity component data as 
        ``{'gz':gzdata, 'gxx':gxxdata, 'gxy':gxydata, ...}``
        If there is no data for a given component, omit the respective key.
        Each g*data is a data dictionary (see bellow)
        
    * residuals
        1D array-like residuals vector
        
    Returns:
    
    * adjusted_data
        Adjusted data in the same format as *data*
        
    The data dictionaries should be as::
    
        {'x':[x1, x2, ...], 'y':[y1, y2, ...], 'z':[z1, z2, ...],
         'value':[data1, data2, ...], 'error':[error1, error2, ...]}
         
    """
    
    global supported_fileds
    
    reduced = residuals.copy()
    
    adjusted = {}
    
    for field in supported_fileds:
        
        if field in data:
            
            adjusted[field] = data[field].copy()
            
            ndata = len(data[field]['value'])
            
            adjusted[field]['value'] = data[field]['value'] - reduced[0:ndata]
    
            reduced = reduced[ndata:]
            
    return adjusted


def _calc_adjusted_vector(estimate):
    """
    Calculate the adjusted data vector produced by a given estimate.    
    """
        
    jacobian = _build_pgrav3d_jacobian(estimate)
    
    adjusted = numpy.dot(jacobian, estimate)
            
    return adjusted


def _build_pgrav3d_jacobian(estimate):
    """Build the Jacobian matrix of the gravity field"""
    
    assert _mesh is not None, "Can't build Jacobian. No mesh defined"
    assert _data is not None, "Can't build Jacobian. No data defined"
    
    global _jacobian, supported_fileds
    
    if _jacobian is None:
        
        start = time.time()
        
        _jacobian = []
        append_row = _jacobian.append
        
        for field in supported_fileds:
            
            if field in _data:
                
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
    
    If *z0* or *power* are set to ``None``, they will be automatically 
    calculated.
    
    Parameters:
    
    * mesh
        Model space discretization mesh (see :func:`fatiando.mesh.prism_mesh`)
    
    * z0
        Compensation depth
    
    * power
        Power of the power law used
    
    * grid_height
        Height of the data grid in meters (only needed if *z0* and *power* are 
        ``None``)
    
    * normalize
        If ``True``, normalize the weights
    
    Returns:
    
    * [z0, power]
        Values used for *z0* and *power*
        
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
            
            kernel = fatiando.grav.prism.gzz(1., cell['x1'], cell['x2'], 
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


def set_bounds(vmin, vmax):
    """
    Set bounds on the density values.
    
    Parameters:
    
    * vmin
        Lowest value the density can assume
    
    * vmax
        Highest value the density can assume
    
    """
    
    log.info("Setting bounds on density values:")
    log.info("  vmin: %g" % (vmin))
    log.info("  vmax: %g" % (vmax))
    
    solvers.set_bounds(vmin, vmax)


def solve(data, mesh, initial=None, damping=0, smoothness=0, curvature=0, 
          sharpness=0, beta=10**(-5), compactness=0, epsilon=10**(-5), 
          max_it=100, lm_start=1, lm_step=10, max_steps=20):    
    """
    Solve the inverse problem for the density using a given data set.
        
    **NOTE**: only uses *max_it*, *lm_start*, *lm_step* and *max_steps* if also 
    using *sharpness*, *compactness* or bounds of the parameter values
    (eg :func:`fatiando.inversion.pgrav3d.set_bounds`) because otherwise the 
    problem is linear.
    
    Parameters:
    
    * data 
        Dictionary with the gravity component data as 
        ``{'gz':gzdata, 'gxx':gxxdata, 'gxy':gxydata, ...}``
        If there is no data for a given component, omit the respective key.
        Each g*data is a profile data dictionary (see bellow)
      
    * mesh
        Model space discretization mesh (see :func:`fatiando.mesh.prism_mesh`)
      
    * initial
        Initial estimate (only used with *sharpness* or *compactness*). 
        If ``None``, will use zero
    
    * damping
        Tikhonov order 0 regularization parameter. Must be >= 0
    
    * smoothness
        Tikhonov order 1 regularization parameter. Must be >= 0
    
    * curvature
        Tikhonov order 2 regularization parameter. Must be >= 0
    
    * sharpness    
        Total Variation regularization parameter. Must be >= 0
    
    * beta
        Small constant used to make Total Variation differentiable. 
        Must be >= 0. The smaller it is, the sharper the solution but also 
        the less stable
            
    * compactness
        Compact regularization parameter. Must be >= 0
      
    * epsilon
        Small constant used in Compact regularization to avoid singularities. 
        Set it small for more compactness, larger for more stability.
    
    * max_it
        Maximum number of iterations 
      
    * lm_start
        Initial Marquardt parameter (ie, step size)
    
    * lm_step 
        Factor by which the Marquardt parameter will be reduced with each 
        successful step
           
    * max_steps
        How many times to try giving a step before exiting

    Returns:
    
    * [estimate, residuals, goals]
        estimate = array-like parameter vector estimated by the inversion.
        residuals = array-like residuals vector.
        goals = list of goal function value per iteration.
        
    The data dictionaries should be as::
    
        {'x':[x1, x2, ...], 'y':[y1, y2, ...], 'z':[z1, z2, ...],
         'value':[data1, data2, ...], 'error':[error1, error2, ...]}
         
    **NOTE**: Use :func:`fatiando.mesh.fill` to put the estimate in a  mesh so 
    you can plot and save it (using ``pickle``).
    """

    for key in data:
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
    solvers._calc_adjustment = _calc_adjusted_vector
    
    solvers.damping = damping
    solvers.smoothness = smoothness
    solvers.curvature = curvature
    solvers.sharpness = sharpness
    solvers.beta = beta
    solvers.compactness = compactness
    solvers.epsilon = epsilon
    
    estimate, residuals, goals = solvers.lm(_data_vector, None, initial, 
                                            lm_start, lm_step, max_steps, 
                                            max_it)

    return estimate, residuals, goals


def get_seed(point, density, mesh):
    """
    Returns as a seed the cell in the mesh that has *point* inside it.
    
    Use this to get the seeds needed by :func:`fatiando.inversion.pgrav3d.grow`
    
    **UNITS**: use SI for all units.
    
    Parameters:
    
    * point
        ``(x,y,z)`` as coordinates of a point
        
    * density
        The density of the seed
        
    * mesh
        Model space discretization mesh (see :func:`fatiando.mesh.prism_mesh`)
        
    Returns:
    
    * seed
        A dictionary with the seed properties:
        ``{'index':index_of_seed, 'density':density, 'cell':cell_in_mesh, 
        'neighbors':[]}``
        
    Raises:
    
    * ValueError
        If *point* is not in the mesh
    
    """
    
    x, y, z = point
    
    seed = None
    
    for i, cell in enumerate(mesh.ravel()):
        
        if (x >= cell['x1'] and x <= cell['x2'] and y >= cell['y1'] and  
            y <= cell['y2'] and z >= cell['z1'] and z <= cell['z2']):
            
            seed = {'index':i, 'density':density, 'cell':cell.copy(),
                    'neighbors':[]}

            seed['cell']['value'] = density
            
            break
        
    if seed is None:
        
        raise ValueError("There is no cell in 'mesh' with 'point' inside it.")
    
    log.info("  seed: %s" % (str(seed)))

    return seed


def _calc_jac_col(index, data, mesh):
    """Calculate a column of the Jacobian matrix"""
            
    global _calculators, supported_fileds
    
    cell = mesh.ravel()[index]
    
    x1 = cell['x1']
    x2 = cell['x2']
    y1 = cell['y1']
    y2 = cell['y2']
    z1 = cell['z1']
    z2 = cell['z2']
    
    column = []
    
    # Can't do 'for field in data' because they need to be appended to the
    # column in a specific order given by 'supported_fields'
    for field in supported_fileds:
                    
        if field in data:
            
            coordinates =  zip(data[field]['x'], data[field]['y'], 
                               data[field]['z'])
            
            function = _calculators[field]
            
            column.extend([function(1., x1, x2, y1, y2, z1, z2, x, y, z)
                           for x, y, z in coordinates])
            
    return numpy.array(column)


def _cell_distance(cell, seed):
    """
    Calculate the distance from cell to seed in number of cells.
    
    Parameters:
    
    *cell
        Dictionary describing the cell (see :func:`fatiando.geometry.prism`)
      
    * seed
        Dictionary describing the seed 
        (see :func:`fatiando.inversion.pgrav3d.get_seed`)
        
    Returns:
    
    * distance
    
    """
                    
    x_distance = abs(cell['x1'] - seed['cell']['x1'])/(cell['x2'] - cell['x1'])
    y_distance = abs(cell['y1'] - seed['cell']['y1'])/(cell['y2'] - cell['y1'])
    z_distance = abs(cell['z1'] - seed['cell']['z1'])/(cell['z2'] - cell['z1'])
    
    distance = max([x_distance, y_distance, z_distance])
    
    return distance


def _radial_distance(cell, seed):
    """
    Calculate the radial distance from cell to seed in number of cells.
    
    Parameters:
    
    *cell
        Dictionary describing the cell (see :func:`fatiando.geometry.prism`)
      
    * seed
        Dictionary describing the seed 
        (see :func:`fatiando.inversion.pgrav3d.get_seed`)
        
    Returns:
    
    * distance
    
    """
    
#    x_distance = abs(cell['x1'] - seed['cell']['x1'])
#    y_distance = abs(cell['y1'] - seed['cell']['y1'])
#    z_distance = abs(cell['z1'] - seed['cell']['z1'])
    
    cellx = 0.5*(cell['x2'] + cell['x1'])
    celly = 0.5*(cell['y2'] + cell['y1'])
    cellz = 0.5*(cell['z2'] + cell['z1'])
    
    seedx = 0.5*(seed['cell']['x2'] + seed['cell']['x1'])    
    seedy = 0.5*(seed['cell']['y2'] + seed['cell']['y1'])    
    seedz = 0.5*(seed['cell']['z2'] + seed['cell']['z1'])
    
    x_distance = abs(cellx - seedx)
    y_distance = abs(celly - seedy)
    z_distance = abs(cellz - seedz)    
    
    distance = math.sqrt(x_distance**2 + y_distance**2 + z_distance**2)
    
    return distance


def _get_reduced_neighbors(param, estimate, seeds, mesh, which=False):
    """
    Get the non-diagonal neighbors of *param* in *mesh*.
    
    Parameters:
    
    * param
        The index of the parameter in the parameter vector
      
    * estimate
        Dictionary with the already appended cells
      
    * seeds
        List of all seeds. Used to check for repeating neighbors
      
    * mesh
        The model space mesh
        
    * which
        If ``True``, return also a list informing which neighbors exist.
        (This is only meant to be used to compute the diagonal neighbors in
         :func:`fatiando.inversion.pgrav3d._get_full_neighbors`)
        
    Returns:
    
    * neighbors
        List with the index of each neighbor in the parameter vector
    
    """
    
    neighbors = []
    
    nz, ny, nx = mesh.shape
    
    append = neighbors.append
    
    # The guy above
    neighbor = param - nx*ny
    above = None
    
    if neighbor > 0:
        
        above = neighbor
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
    
    # The guy bellow
    neighbor = param + nx*ny
    bellow = None
    
    if neighbor < mesh.size:
        
        bellow = neighbor
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
    
    # The guy in front
    neighbor = param + 1
    front = None
    
    if param%nx < nx - 1:
        
        front = neighbor
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
    
    # The guy in the back
    neighbor = param - 1
    back = None
    
    if param%nx != 0:
        
        back = neighbor
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
    
    # The guy to the left
    neighbor = param + nx
    left = None
    
    if param%(nx*ny) < nx*(ny - 1):
        
        left = neighbor
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
    
    # The guy to the right
    neighbor = param - nx
    right = None
    
    if param%(nx*ny) >= nx:
        
        right = neighbor
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
            
    if which:
                
        return neighbors, [above, bellow, front, back, left, right]
    
    else:
 
        return neighbors
            
            
def _get_full_neighbors(param, estimate, seeds, mesh):
    """
    Get all the neighbors of *param* in *mesh*, including the diagonals.
    
    Parameters:
    
    * param
        The index of the parameter in the parameter vector
      
    * estimate
        Dictionary with the already appended cells
      
    * seeds
        List of all seeds. Used to check for repeating neighbors
      
    * mesh
        The model space mesh
        
    Returns:
    
    * neighbors
        List with the index of each neighbor in the parameter vector
    
    """
    
    nz, ny, nx = mesh.shape
    
    neighbors, which = _get_reduced_neighbors(param, estimate, seeds, mesh, 
                                              which=True)
    
    append = neighbors.append
    
    above, bellow, front, back, left, right = which

    # The diagonals
         
    if front is not None and left is not None:
        
        neighbor = left + 1
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
    
    if front is not None and right is not None:
        
        neighbor = right + 1
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
            
    if back is not None and left is not None:
        
        neighbor = left - 1
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
            
    if back is not None and right is not None:
    
        neighbor = right - 1
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
            
    if above is not None and left is not None:
        
        neighbor = above + nx
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
            
    if above is not None and right is not None:
        
        neighbor = above - nx
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
            
    if above is not None and front is not None:
        
        neighbor = above + 1
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
            
    if above is not None and back is not None:
        
        neighbor = above - 1
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
            
    if above is not None and front is not None and left is not None:
        
        neighbor = above + nx + 1
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
            
    if above is not None and front is not None and right is not None:
        
        neighbor = above - nx + 1   
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)

    if above is not None and back is not None and left is not None:
        
        neighbor = above + nx - 1 
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
    
    if above is not None and back is not None and right is not None:
        
        neighbor = above - nx - 1 
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
            
    if bellow is not None and left is not None:
        
        neighbor = bellow + nx
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
            
    if bellow is not None and right is not None:
        
        neighbor = bellow - nx
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
            
    if bellow is not None and front is not None:
        
        neighbor = bellow + 1
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
            
    if bellow is not None and back is not None:
        
        neighbor = bellow - 1
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
            
    if bellow is not None and front is not None and left is not None:
        
        neighbor = bellow + nx + 1
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
            
    if bellow is not None and front is not None and right is not None:
        
        neighbor = bellow - nx + 1
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
        
    if bellow is not None and back is not None and left is not None:
        
        neighbor =  bellow + nx - 1
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
            
    if bellow is not None and back is not None and right is not None:
        
        neighbor = bellow - nx - 1
            
        # Need to check if neighbor is not in any seed's neighbors and has not
        # been marked
        is_neighbor = [neighbor in seed['neighbors'] for seed in seeds]
        is_marked = neighbor in estimate
        
        if True not in is_neighbor and not is_marked:
        
            append(neighbor)
 
    return neighbors


def _dump_jac_col(index, column, outfile):
    """
    Dump a column of the Jacobian matrix to a file in a zip archive.
    
    The name of the file will be the *index* of the column.
    
    Parameters:
    
    * index
        Integer index of the column in the Jacobian matrix
        
    * column
        1D array-like column to be dumped
        
    * outfile
        Open ZipFile instance with writing privileges (see module ``zipfile``)
        
    """
    
    if outfile is not None:
    
        fname = str(index)
        
        # If getinfo doesn't raise an exception, it's because the file was 
        # already in the archive, so don't add it again
        try:
            
            outfile.getinfo(fname)
            
        except KeyError:      
        
            stream = StringIO.StringIO()
            
            pylab.savetxt(stream, column)
            
            outfile.writestr(fname, stream.getvalue())    
    
    
def _get_jac_col(index, data, mesh, infile):
    """
    Calculate a column of the Jacobina matrix or load it from a zip archive.
        
    Parameters:
    
    * index
        Integer index of the column in the Jacobian matrix
                
    * data 
        Dictionary with the gravity component data as 
        ``{'gz':gzdata, 'gxx':gxxdata, 'gxy':gxydata, ...}``
        If there is no data for a given component, omit the respective key.
        Each g*data is a data dictionary
        
    * mesh
        Model space discretization mesh (see :func:`fatiando.mesh.prism_mesh`)
        
    * infile
        Open ZipFile instance with reading privileges (see module ``zipfile``)
    
    Returns:
    
    * column
        1D array-like column of the Jacobian matrix
        
    """
    
    if infile is None:
        
        column = _calc_jac_col(index, data, mesh)
        
    else:
    
        fname = str(index)
            
        # If getinfo doesn't raise an exception, it's because the file is  
        # already in the archive
        try:
            
            col_string = infile.read(fname)
            
            stream = StringIO.StringIO(col_string)
    
            column = pylab.loadtxt(stream)
            
        except KeyError:
        
            column = _calc_jac_col(index, data, mesh)
        
    return column


def _get_data_weights(data, seeds):
    """
    """
    
    global supported_fileds
    
    weights = []
    
    for field in supported_fileds:
        
        if field in data:
            
            sizex = data[field]['x'].max() - data[field]['x'].min()
            sizey = data[field]['y'].max() - data[field]['y'].min()
            
            size = numpy.sqrt(sizex**2 + sizey**2)
                        
            for x, y in zip(data[field]['x'], data[field]['y']):
                
                distance = None
                
                for seed in seeds:
                    
                    xseed = 0.5*(seed['cell']['x2'] + seed['cell']['x1'])
                    
                    yseed = 0.5*(seed['cell']['y2'] + seed['cell']['y1'])
    
                    dist = numpy.sqrt((x - xseed)**2 + (y - yseed)**2)
                    
                    if distance is None or dist < distance:
                        
                        distance = dist
                        
                # Avoid zero division erros
                if distance < 10**(-15):
                    
                    distance = 10**(-15)
                    
                weights.append(size/distance)
                
    weights = numpy.array(weights)
                
    # Normalize the weights
    weights = weights/weights.max()
    
    return weights                    
                

def grow(data, mesh, seeds, compactness, power=5, threshold=10**(-4), norm=2,
         jacobian_file=None, neighbor_type='reduced', distance_type='cell'):
    """
    Invert by growing the solution around given seeds.
    
    Parameters:
        
    * data 
        Dictionary with the gravity component data as 
        ``{'gz':gzdata, 'gxx':gxxdata, 'gxy':gxydata, ...}``
        If there is no data for a given component, omit the respective key.
        Each g*data is a data dictionary (see bellow)
      
    * mesh
        Model space discretization mesh (see :func:`fatiando.mesh.prism_mesh`)
      
    * seeds
        list of seeds (to make a seed, see 
        :func:`fatiando.inversion.pgrav3d.get_seed`)
      
    * compactness
        How compact the solution should be around the seeds (regularization
        parameter). Must to be >= 0
           
    * power
        Power to which the distances are raised in the *compactness* weights
        
    * threshold
        Minimum decimal percentage reduction of the RMS required to grow
      
    * norm
        What norm of the residuals to use for the misfit. Should be an integer:
        
        * ``1``
            The :math:`l_1` norm of the residuals. Used for robustness against
            outliers in the data set.
            
        * ``2``
            The :math:`l_2` norm of the residuals. Used for a *Least* *Squares*
            fit of the data. (DEFAULT)
        
    * jacobian_file
        Name of a .zip file used to store the Jacobian matrix. If file already
        exists, will try to load the Jacobian columns from the file. Otherwise,
        will create a file and save the Jacobian columns to it.
        
    * neighbor_type
        The neighbors are the available cells around a seed that can be added to
        the estimate. Can be either:
    
        * ``'reduced'``
            Only use the front, back, left, right, top and bottom neighbors of a
            cell. Use this to save computation time and memory.
            
        * ``'full'``
            Also use the diagonal neighbors. 
        
    * distance_type
        The distances between a cell and the seeds is used as parameter weights
        to enforce compactness. Can be either:
        
        * ``'radial'``
            Use the distance from the center of the cell to the center of the 
            seed. Use this to make the estimated bodies rounder.
            
        * ``'cell'``
            Use the distance in number of cells in the mesh from the cell to the
            seed. Use this to make the estimated bodies more rectangular. 
      
    Returns:
    
    * [estimate, residuals, goals, misfits]:
        estimate = array-like parameter vector estimated by the inversion.
        residuals = array-like residuals vector.
        goals = list of goal function value per iteration.
        misfits = list of misfit value per iteration.
            
    Raises:
    
    * AttributeError
        If seeds are too close (less than 2 cells appart)
        
    The data dictionaries should be as::
    
        {'x':[x1, x2, ...], 'y':[y1, y2, ...], 'z':[z1, z2, ...],
         'value':[data1, data2, ...], 'error':[error1, error2, ...]}
        
    """    

    # Initial sanity checks
    for key in data:
        assert key in ['gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz'], \
            "Invalid gravity component (data key): %s" % (key)
            
    assert neighbor_type in ['full', 'reduced'], \
        "Invalid neighbor type '%s'" % (neighbor_type)
    
    assert distance_type in ['radial', 'cell'], \
        "Invalid distance type '%s'" % (distance_type)
        
    assert norm == 1 or norm == 2, "Invalid norm '%s'." % (str(norm))
            
    # Set the neighbor and distance types
    if distance_type == 'radial':
        
        get_distance = _radial_distance
        
    elif distance_type == 'cell':
        
        get_distance = _cell_distance
        
    if neighbor_type == 'full':
        
        get_neighbors = _get_full_neighbors
        
    elif neighbor_type == 'reduced':
        
        get_neighbors = _get_reduced_neighbors
        
    # Open the zip file containing the Jacobian columns (or create a new one)    
    if jacobian_file is None:
        
        j_zip = None
        
    else:
        
        if os.path.exists(jacobian_file):
        
            j_zip = zipfile.ZipFile(jacobian_file, 'a')
            
        else:
            
            j_zip = zipfile.ZipFile(jacobian_file, 'w')
    
    
    # Initialize the residuals
    residuals = extract_data_vector(data)
    
    # Weight the residuals with the absolute value of the data to prioritize
    # lower residuals over the extrema of the data
#    weights =  _get_data_weights(data, seeds)
    weights = numpy.ones_like(residuals)
            
    # Define a lambda function to compute the norm of the residuals
    if norm == 1:
        
        calc_norm = lambda r: (weights*numpy.abs(r)).sum()
        
    elif norm == 2:
        
        calc_norm = lambda r: (weights*(r**2)).sum()
    
    # Initialize the estimate          
    estimate = {}
    
    for seed in seeds:
        
        estimate[seed['index']] = seed['density']
        
        tmp_jac_col = _get_jac_col(seed['index'], data, mesh, j_zip)
    
        residuals -= seed['density']*tmp_jac_col
        
        _dump_jac_col(seed['index'], tmp_jac_col, j_zip)
    
        del tmp_jac_col
        
    # Initialize the neighbor and distance lists and the Jacobian matrix
    # (couldn't do this before because I need the estimate filled with the seeds 
    # not to mark them as neighbors)
    
    jacobian = {}
    
    for seed in seeds:
        
        seed['distances'] = []
        
        new_neighbors = get_neighbors(seed['index'], estimate, seeds, mesh)
        
        for neighbor in new_neighbors:
            
            jacobian[neighbor] = _get_jac_col(neighbor, data, mesh, j_zip)
            
            distance = get_distance(mesh.ravel()[neighbor], seed)            
            
            seed['distances'].append(distance)
            
        seed['neighbors'].extend(new_neighbors)
    
    misfits = [calc_norm(residuals)]
    
    # Since there are only the seeds in the estimate, the compactness 
    # regularizer is 0
    regularizer = 0.
    
    goals = [misfits[-1] + regularizer]
    
    log.info("Growing density model:")
    log.info("  parameters = %d" % (mesh.size))
    log.info("  data = %d" % (len(residuals)))
    log.info("  compactness = %g" % (compactness))
    log.info("  power = %g" % (power))
    log.info("  threshold = %g" % (threshold))
    log.info("  norm = l%d" % (norm))
    log.info("  neighbor type = %s" % (neighbor_type))
    log.info("  distance type = %s" % (distance_type))
    log.info("  Jacobian matrix file = %s" % (jacobian_file))
    log.info("  initial misfit = %g" % (misfits[-1]))
    log.info("  initial total goal function = %g" % (misfits[-1]))
    
    total_start = time.time()
        
    for iteration in xrange(mesh.size - len(seeds)):
        
        start = time.time()
        
        log.info("  it %d:" % (iteration + 1))
        
        grew = False
            
        # Try to grow each seed by one using the goal function as a criterium
        # NOTE: The order of the seeds affects the growing (the goal changes
        # when a seed grows)!
        for seed_num, seed in enumerate(seeds):
    
            density = seed['density']
    
            best_goal = None
            best_reg = None
            best_misfit = None
            best_neighbor = None
            
            for neighbor, distance in zip(seed['neighbors'], seed['distances']):
                
                new_residuals = residuals - density*jacobian[neighbor]
                
                misfit = calc_norm(new_residuals)
    
                regularizer_increment = compactness*(distance**power)
                                
                goal = misfit + regularizer + regularizer_increment
                
                # Reducing the misfit is mandatory while also looking for the one
                # that minimizes the total goal the most
                if (misfit < misfits[-1] and 
                    abs(misfit - misfits[-1])/misfits[-1] >= threshold):
                    
                    if best_goal is None or goal < best_goal:
                    
                        best_neighbor = neighbor
                        best_goal = goal
                        best_misfit = misfit
                        best_reg = regularizer + regularizer_increment
                        
            if best_neighbor is not None:
                    
                grew = True
                
                estimate[best_neighbor] = density
                
                residuals -= density*jacobian[best_neighbor]
                
                regularizer = best_reg
                                                    
                misfits.append(best_misfit)
                
                goals.append(best_goal)
                
                # Remove the chosen one from the list of available neighbors and
                # dump it's Jacobian column to a file to save memory
                best_index = seed['neighbors'].index(best_neighbor)                
                seed['neighbors'].pop(best_index)
                seed['distances'].pop(best_index)
                
                _dump_jac_col(best_neighbor, jacobian[best_neighbor], j_zip)
        
                del jacobian[best_neighbor]
                    
                # Update the neighbors, distances and Jacobian
                new_neighbors = get_neighbors(best_neighbor, estimate, seeds, 
                                              mesh)
        
                for neighbor in new_neighbors:
                    
                    jacobian[neighbor] = _get_jac_col(neighbor, data, mesh, 
                                                      j_zip)
                    
                    distance = get_distance(mesh.ravel()[neighbor], seed)
                    
                    seed['distances'].append(distance)
                    
                seed['neighbors'].extend(new_neighbors)
                                
                log.info(''.join(['    append to seed %d:' % (seed_num + 1),
                                  ' size=%d' % (len(estimate)),
                                  ' neighbors=%d' % (len(seed['neighbors'])),
                                  ' MISFIT=%g' % (best_misfit),
                                  ' COMPACTNESS=%g' % (best_reg),
                                  ' GOAL=%g' % (best_goal)]))
                          
        if not grew:
                                
            log.warning("    Exited because couldn't grow.")
            
            break
                                
        end = time.time()
        
        log.info("    time: %g s" % (end - start))
    
    log.info("  Size of estimate: %d cells" % (len(estimate)))
            
    # Finish dumping the Jacobian to a file
    if j_zip is not None:
        
        log.info("  Dumping the Jacobian to file '%s'" % (jacobian_file))
        
        for i in jacobian:
            
            _dump_jac_col(i, jacobian[i], j_zip)
            
        j_zip.close()
        
    del jacobian
    
    # Fill the estimate with zeros
    log.info("  Filling estimate with zeros...")
    
    estimate_vector = numpy.zeros(mesh.size)
    
    for i in estimate:
    
        estimate_vector[i] = estimate[i]
        
    total_end = time.time()
    
    log.info("  Total inversion time: %g s" % (total_end - total_start))

    return estimate_vector, residuals, misfits, goals