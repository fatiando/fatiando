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
Finite Differences (FD) solvers for the 1D heat diffusion equation.

Functions:

* :func:`fatiando.heat.diffusionfd1d.fixed_bc`
    Set fixed boundary conditions.

* :func:`fatiando.heat.diffusionfd1d.free_bc`
    Set free edges (zero derivative) boundary conditions.
    
* :func:`fatiando.heat.diffusionfd1d.timestep`
    Perform a single time step.

* :func:`fatiando.heat.diffusionfd1d.run`
    Run many timesteps of the simulation.
    
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 29-Sep-2010'


import logging

import fatiando
import fatiando.heat._diffusionfd1d as diffusionfd1d_ext
   

# Add the default handler (a null handler) to the logger to ensure that
# it won't print verbose if the program calling them doesn't want it
log = logging.getLogger('fatiando.heat.diffusionfd1d')
log.addHandler(fatiando.default_log_handler)

    
def fixed_bc(start_val, end_val):
    """
    Set fixed boundary conditions.
    
    Parameters:
      
    * start_val
        Value to fix at the starting position
      
    * end_val
        Value to fix at the ending position
      
    Returns:
    
    * [start_bc, end_bc]
        Callable boundary conditions to pass to 
        :func:`fatiando.heat.diffusionfd1d.run`
         
    """
    
    def start_bc(temps):
        
        temps[0] = start_val
        
    def end_bc(temps):
        
        temps[-1] = end_val

    return start_bc, end_bc
    
    
def free_bc():
    """
    Set free edges (zero derivative) boundary conditions.
          
    Returns:
    
    * [start_bc, end_bc]
        Callable boundary conditions to pass to 
        :func:`fatiando.heat.diffusionfd1d.run`
         
    """
    
    def start_bc(temps):
        
        temps[0] = temps[1]
        
    def end_bc(temps):
        
        temps[-1] = temps[-2]
        
    return start_bc, end_bc
    
    
def timestep(temp, deltax, deltat, diffusivity, start_bc, end_bc):
    """
    Perform a single time step.
    
    For the boundary conditions see:
    
    :func:`fatiando.heat.diffusionfd1d.free_bc`
    
    :func:`fatiando.heat.diffusionfd1d.fixed_bc`
    
    Parameters:
    
    * temp
        1D array-like temperature on each FD node
      
    * deltax
        Spacing between the nodes
      
    * deltat 
        Time step
    
    * diffusivity
        1D array-like thermal diffusivity on each FD node
    
    * start_bc
        Callable boundary condition at the starting point
    
    * end_bc
        Callable boundary condition at the ending point
            
    Returns:
    
    * temp
        1D array-like temperature on each FD node at the next time
        
    """
    
    temp_tp1 = diffusionfd1d_ext.timestep_explicit(temp, diffusivity, deltat, 
                                                   deltax)
        
    start_bc(temp_tp1)
    
    end_bc(temp_tp1)
    
    return temp_tp1

    
def run(deltax, deltat, diffusivity, initial, start_bc, end_bc, ntimes):
    """
    Run many timesteps of the simulation.
    
    For the boundary conditions see:
    
    :func:`fatiando.heat.diffusionfd1d.free_bc`
    
    :func:`fatiando.heat.diffusionfd1d.fixed_bc`
    
    Parameters:
      
    * deltax
        Spacing between the nodes
    
    * deltat
        Time step
    
    * diffusivity
        1D array-like thermal diffusivity on each FD node
    
    * initial
        1D array-like temperature on each FD node
    
    * start_bc
        Callable boundary condition at the starting point
    
    * end_bc
        Callable boundary condition at the ending point
    
    * ntimes
        Number of time steps to run
      
    Returns:
    
    * temps
        1D array-like temperature on each FD node at the end of the run
        
    """
    
    next = list(initial)
        
    start_bc(next)
    
    end_bc(next)
        
    for time in xrange(ntimes):
        
        prev = next
        
        next = timestep(prev, deltax, deltat, diffusivity, start_bc, end_bc)
        
    return next
    