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
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 29-Sep-2010'


from fatiando.heat._diffusionfd import timestep1d as fortran_timestep
   
    
    
def fixed_bc(start_val, end_val):
    """
    Set fixed boundary conditions.
    
    Parameters:
      
      start_val: value to fix at the starting position
      
      end_val: value to fix at the ending position
      
    Returns:
    
      [start_bc, end_bc]: callable boundary conditions to pass to 'run' 
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
    
      [start_bc, end_bc]: callable boundary conditions to pass to 'run' 
    """
    
    def start_bc(temps):
        
        temps[0] = temps[1]
        
    def end_bc(temps):
        
        temps[-1] = temps[-2]
        
    return start_bc, end_bc
    
    
def timestep(temp, deltax, deltat, diffusivity, start_bc, end_bc):
    """
    Run a single time step of the Finite Differences simulation of the 1D heat 
    diffusion equation
    
    Parameters:
    
      temp: 1D array-like temperature on each FD node
      
      deltax: spacing between the nodes
      
      deltat: time step
      
      diffusivity: 1D array-like thermal diffusivity on each FD node
      
      start_bc: callable boundary condition at the starting point
      
      end_bc: callable boundary condition at the ending point
            
    Returns:
    
      temp: 1D array-like temperature on each FD node at the next time
    """
    
    temp_tp1 = fortran_timestep(temp, diffusivity, deltat, deltax)
        
    start_bc(temp_tp1)
    
    end_bc(temp_tp1)
    
    return temp_tp1

    
def run(deltax, deltat, diffusivity, initial, start_bc, end_bc, ntimes):
    """
    Run the Finite Differences simulation of the 1D heat diffusion equation
    
    Parameters:
      
      deltax: spacing between the nodes
      
      deltat: time step
      
      diffusivity: 1D array-like thermal diffusivity on each FD node
    
      initial: 1D array-like temperature on each FD node
      
      start_bc: callable boundary condition at the starting point
      
      end_bc: callable boundary condition at the ending point
      
      ntimes: number of time steps to run
      
    Returns:
    
      temps: 1D array-like temperature on each FD node at the end of the run
    """
    
    next = list(initial)
        
    start_bc(next)
    
    end_bc(next)
        
    for time in xrange(ntimes):
        
        prev = next
        
        next = timestep(prev, deltax, deltat, diffusivity, start_bc, end_bc)
        
    return next
    