"""
WaveFD:
    Finite differences solvers for the elastic wave equation.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 29-June-2010'


import logging
import math
import time

import numpy
import pylab

from fatiando.utils import contaminate
from fatiando.directmodels.seismo import wavefd_ext
import fatiando


logger = logging.getLogger('wavefd')       
logger.setLevel(logging.DEBUG)
logger.addHandler(fatiando.default_log_handler)

logger = logging.getLogger('wavesource')       
logger.setLevel(logging.DEBUG)
logger.addHandler(fatiando.default_log_handler)



class SinSQWaveSource():
    """sin^2(t) wave source"""

    def __init__(self, amplitude, period, duration, offset, index):
        
        self._period = period
        
        self.amplitude = amplitude
        
        self._duration = duration
        
        self._offset = offset
        
        self._index = index
    
        self._log = logging.getLogger('wavesource')
        

    def at(self, time):
        """What the value of the source is at a given time"""

        return self.amplitude*math.sin(math.pi*(time + self._offset)/ \
                                       self._period)**2
    

    def active(self, time):
        """True if the source is still active at 'time', False otherwise."""
        
        if time > self._duration:
            
            return False
        
        else:
            
            return True
        
        
    def pos(self):
        """Grid index of where the source is located"""
        
        return self._index



class WaveFD1D():
    """
    Finite differences solver for the 1D elastic wave.
    """
    
    def __init__(self, x1, x2, num_nodes, delta_t, velocities, source, \
                 left_bc='free', right_bc='free'):
                
        # Grid parameters
        self._x1 = x1
        self._x2 = x2
        self._deltax = (x2 - x1)/float(num_nodes - 1.)
        self._num_nodes = num_nodes
        self._vel = velocities
        self._u_t = None
        self._u_tp1 = None
        self._u_tm1 = None
        
        # Boundary conditions
        self._left_bc = left_bc
        self._right_bc = right_bc    
        
        self._source = source
        
        self._deltat = delta_t
        self._time = 0
    
        self._log = logging.getLogger('wavesource')
        
    
    def _set_bc(self):
        """
        Impose the boundary conditions
        """
        
        if self._left_bc == 'free':
            
            self._u_tp1[0] = self._u_tp1[1]
            
        if self._left_bc == 'fixed':
            
            self._u_tp1[0] = 0
            
        if self._right_bc == 'free':            
            
            self._u_tp1[-1] = self._u_tp1[-2]
            
        if self._right_bc == 'fixed':
            
            self._u_tp1[-1] = 0
                    
        
    def timestep(self):
        """
        Perform a time step.
        """
        self._first_timestep()
        
        
    def _first_timestep(self):
        """
        Do the first time step (put zeros on all 'u's and start the source).
        Sets the timestep to _second_timestep
        """
        
        
        
        if self._time == 0:
            
            self._u_tp1 = numpy.zeros(self._num_nodes).tolist()
        
        elif self._time == self._deltat:
            
            self._u_t = self._u_tp1
            
            self._u_tp1 = numpy.zeros(self._num_nodes).tolist()
                            
        else:
                        
            self._u_tm1 = self._u_t
            
            self._u_t = self._u_tp1
                        
            self._u_tp1 = wavefd_ext.timestep1d(self._deltax, self._deltat, \
                                                self._u_tm1, \
                                                self._u_t, \
                                                self._vel.tolist())
                        
        self._set_bc()

        if self._source.active(self._time):
            
            self._u_tp1[self._source.pos()] = self._source.at(self._time)
                                        
        self._time += self._deltat
        
    
    def plot(self, title=""):
        """
        Plot the current amplitudes.
        """
        
        pylab.figure()
        pylab.title(title + " time: %g" % (self._time))
        
        x = numpy.arange(self._x1, self._x2 + self._deltax, self._deltax)
        
        # Sometimes there is trouble with the number of points in x
        # This is because of rounding self._deltax, making there be one extra
        # point in the end
        if len(x) > self._num_nodes:
            
            x = x[:-1]
        
        pylab.plot(x, self._u_tp1, '-k')
        pylab.xlim(self._x1, self._x2)
        pylab.ylim(-abs(self._source.amplitude), abs(self._source.amplitude))
        
        
        
        