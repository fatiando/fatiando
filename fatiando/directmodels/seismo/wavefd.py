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

    def __init__(self, amplitude, period, duration, offset, index=0):
        
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
        
    
    def move(self, index):
        """
        Move the source to the given index
        """
        
        self._index = index
        
        
    def pos(self):
        """Grid index of where the source is located"""
        
        return self._index
    
    
    def copy(self):
        """
        Return a copy of this class
        """

        return SinSQWaveSource(amplitude=self.amplitude, period=self._period, \
                               duration=self._duration, offset=self._offset, \
                               index=self._index)



class WaveFD1D():
    """
    Finite differences solver for the 1D elastic wave.
    """
    
    def __init__(self, x1, x2, num_nodes, delta_t, velocities, source, \
                 left_bc='free', right_bc='free'):
                
        # Grid parameters
        self._x1 = x1
        self._x2 = x2
        self._deltax = (x2 - x1)/float(num_nodes)
        self._num_nodes = num_nodes
        self._vel = velocities
        self._u_t = None
        self._u_tp1 = None
        self._u_tm1 = None
        
        # Geophones to record the event
        self._geophones_index = None        
        self._geophones = None
        
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
            
            
    def _record(self):
        """
        If there are any geophones, record current timestep in them.
        """
        
        if self._geophones != None:
            
            for i in xrange(len(self._geophones)):
                
                self._geophones[i].append([self._time, \
                                    self._u_tp1[self._geophones_index[i]]])
                
                
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
            
        self._u_tp1 = numpy.zeros(self._num_nodes).tolist()

        self._set_bc()

        if self._source.active(self._time):
            
            self._u_tp1[self._source.pos()] = self._source.at(self._time)
                                        
        self._time += self._deltat

        self.timestep = self._second_timestep
        
        self._record()
        
    
    def _second_timestep(self):
        """
        Do the second time step.
        """        
        
        self._u_t = self._u_tp1
        
        self._u_tp1 = numpy.zeros(self._num_nodes).tolist()                

        self._set_bc()

        if self._source.active(self._time):
            
            self._u_tp1[self._source.pos()] = self._source.at(self._time)
                                        
        self._time += self._deltat

        self.timestep = self._normal_timestep
        
        self._record()
        
        
    def _normal_timestep(self):
        """
        Do the normal timestep using the C functions.
        """
                     
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
        
        self._record()
        
        
    def set_geophones(self, offset, deltag, num):
        """
        Set the position of the geophones at the desired grid indexes.
            
        Parameters:
            
            offset: the offset from source to 1st geophone
            
            deltag: distance between geophones
            
            num: how many geophones to use
        """
        
        first = self._source.pos() + int(offset/self._deltax)
        
        deltai = int(deltag/self._deltax)
        
        self._geophones_index = []
        
        self._geophones = []
        
        for i in xrange(num):
            
            self._geophones_index.append(first + i*deltai)
            
            self._geophones.append([])
            
            
    def get_seismogram(self, index):
        """
        Returns a seismogram as a numpy array. 1st column is time, 2nd amplitude 
        """
        
        return numpy.array(self._geophones[index])
            
    
    def plot(self, title="", velocity=False, seismogram=False, tmax=None, \
             xmin=None, xmax=None, exaggerate=1000):
        """
        Plot the current amplitudes.
        
        Parameters:
        
            title: Title of the figure. Will append 'time: %g' to it
            
            velocity: if True, plot the velocities as well
            
            seismogram: if True, plot the seismograms as well
            
            tmax: maximum time range for the seismogram
            
            xmin, xmax: if any, limits of the x coordinate
            
            exaggerate: how much to exaggerate the traces in the seismogram
        """
        
        pylab.figure()
            
        pylab.suptitle(title + " time: %g" % (self._time))
            
        pylab.subplots_adjust(left=0.15)
        
        num_plots = 1
        
        if seismogram:
            
            num_plots += 1
            
        if velocity:
            
            num_plots += 1
        
        if seismogram:
                    
            pylab.subplot(num_plots,1,1)
            
            for i in xrange(len(self._geophones)):
                
                times, amplitudes = numpy.array(self._geophones[i]).T
                
                position = self._x1 + self._deltax*self._geophones_index[i]
                
                x = position + exaggerate*amplitudes
                
                pylab.plot(x, times, '-k') 
                            
            pylab.ylabel("Time")
            
            if tmax != None:
                            
                pylab.ylim(tmax, 0)
            
            else:
                
                pylab.ylim(self._time, 0)
                
            if xmin != None and xmax != None:
                
                pylab.xlim(xmin, xmax) 
                
            else:
                
                pylab.xlim(self._x1, self._x2)
            
            pylab.subplot(num_plots,1,2)
            
        x = numpy.arange(self._x1, self._x2 + self._deltax, self._deltax)
        
        # Sometimes there is trouble with the number of points in x
        # This is because of rounding self._deltax, making there be one extra
        # point in the end
        if len(x) > self._num_nodes:
            
            x = x[:-1]        
        
        pylab.plot(x, self._u_tp1, '-k')
        
        if self._geophones != None:
        
            seismo_x = self._x1 + \
                self._deltax*numpy.array(self._geophones_index)
                
            seismo_y = numpy.zeros_like(seismo_x)
            
            pylab.plot(seismo_x, seismo_y, 'vb', label='Geophone')
                
        pylab.ylim(-abs(2*self._source.amplitude), \
                   abs(2*self._source.amplitude))
        
        pylab.ylabel("Amplitude")
                
        if xmin != None and xmax != None:
            
            pylab.xlim(xmin, xmax) 
            
        else:
            
            pylab.xlim(self._x1, self._x2)
                    
        if velocity:
        
            pylab.subplot(num_plots, 1, 3)
                                
            pylab.plot(x, self._vel, '-k')
                        
            pylab.ylim(0.9*min(self._vel), 1.1*max(self._vel))
                        
            pylab.ylabel("Velocity")
                
            if xmin != None and xmax != None:
                
                pylab.xlim(xmin, xmax) 
                
            else:
                
                pylab.xlim(self._x1, self._x2)
        
        pylab.xlabel("Position")
            
            
    def plot_seismograms(self, exaggerate=5000):
        """
        Plot the recorded seismograms.
            
        Parameters:
        
            exaggerate: how much to exaggerate the traces in the seismogram
        """
        
        pylab.figure()
        pylab.title("Seismograms")
        
        for i in xrange(len(self._geophones)):
            
            times, amplitudes = numpy.array(self._geophones[i]).T
            
            position = self._x1 + self._deltax*self._geophones_index[i]
            
            x = position + exaggerate*amplitudes
            
            pylab.plot(x, times, '-k') 
            
        pylab.xlabel("Position")
        
        pylab.ylabel("Time")
            
        pylab.xlim(self._x1, self._x2)
        
        pylab.ylim(self._time, 0)
        
    
    def plot_velocity(self, title="Velocity structure"):
        """
        Plot the velocities of the grid.
        """       
        
        x = numpy.arange(self._x1, self._x2 + self._deltax, self._deltax)
        
        # Sometimes there is trouble with the number of points in x
        # This is because of rounding self._deltax, making there be one extra
        # point in the end
        if len(x) > self._num_nodes:
            
            x = x[:-1]        
        
        pylab.figure()
        
        pylab.title(title)
        
        pylab.plot(x, self._vel, '-k')
        
        pylab.xlim(self._x1, self._x2)
        
        pylab.ylim(0.9*min(self._vel), 1.1*max(self._vel))
        
        pylab.xlabel("Position")
        
        pylab.ylabel("Velocity")
            