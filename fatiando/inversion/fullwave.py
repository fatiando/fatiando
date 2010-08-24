"""
FullWave:
    Full waveform inversion using Finite Difference solvers.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 05-Jul-2010'

import time
import logging
import math

import pylab
import numpy

import fatiando
from fatiando.inversion.lmsolver import LMSolver
from fatiando.directmodels.seismo import wavefd
from fatiando.data.seismo import Seismogram


logger = logging.getLogger('FullWave1D')       
logger.setLevel(logging.DEBUG)
logger.addHandler(fatiando.default_log_handler)


class FullWave1D(LMSolver):
    """
    1D Full waveform inversion of velocities. 
    """
    
    def __init__(self, xmax, nx, num_nodes, seismogram, source):
        """
        Parameters:
        
            xmax: size of the model space [0:xmax]
            
            nx: how many parts to discretize the model space
            
            num_nodes: number of nodes used in the finite differences simulation
                       of the wave propagation
            
            seismogram: instance of fatiando.data.seismo.Seismogram with the 
                        data
                        
            source: source of the wave that will be used in the FD simulation
                    instance of one of the source classes found in
                    fatiando.directmodels.seismo.wavefd                    
        """
        
        LMSolver.__init__(self)
        
        # Model space parameters
        self._xmax = float(xmax)
        self._nx = nx
        self._nparams = nx
        
        # Data parameters
        self._seismogram = seismogram
        
        # Simulation parameters
        self._num_nodes = num_nodes
        self._source = source
                
        self._log = logging.getLogger('FullWave1D')
        
        self._log.info("Model space discretization: %d parameters" % (nx))
                
    
    def _build_jacobian(self, estimate):
        """
        Make the Jacobian matrix of the function of the parameters.
        'estimate' is the the point in the parameter space where the Jacobian
        will be evaluated.
        """
#        
#        # Expand the simulation space to avoid edge effects. So I need to 
#        # move the source in the grid in order for it to stay in the same place
#        source = self._source.copy()
#        
#        source.move(source.pos() + self._num_nodes)
        
        dvel = 50
        
        deltat = self._seismogram.deltat
        
        offset = self._seismogram.offset
        
        deltag = self._seismogram.deltag
        
        num_gp = self._seismogram.num_geophones
        
        tmax = len(self._seismogram)*deltat
        
        # Build the velocity array to match the model space discretization
        velocities = numpy.empty(self._num_nodes)
        
        node_dx = self._xmax/(self._num_nodes - 1)
        
        model_dx = self._xmax/(self._nx)
        
        nodes_per_cell = int(model_dx/node_dx) + 1
        
        for i in xrange(self._nparams):
            
            index = i*nodes_per_cell
            
            velocities[index:index + nodes_per_cell] = estimate[i]*\
                                                    numpy.ones(nodes_per_cell)

        seismo2 = Seismogram()
        
        seismo1 = Seismogram()
        
        seismo2._log.setLevel(logging.WARNING)
        
        seismo1._log.setLevel(logging.WARNING)
        
        # Start building the Jacobian
        jacobian = []
        
        for i in xrange(self._nparams):
            
            index = i*nodes_per_cell
            
            vel2 = velocities.copy()
            
            vel1 = velocities.copy()
            
            vel2[index:index + nodes_per_cell] = (estimate[i] + dvel)* \
                                                  numpy.ones(nodes_per_cell)
            
            vel1[index:index + nodes_per_cell] = (estimate[i] - dvel)* \
                                                  numpy.ones(nodes_per_cell)

            seismo2.synthetic_1d(offset=offset, deltag=deltag, num_gp=num_gp, \
                                 xmax=self._xmax, num_nodes=self._num_nodes, \
                                 source=self._source, velocities=vel2, \
                                 deltat=deltat, tmax=tmax, stddev=0)

            seismo1.synthetic_1d(offset=offset, deltag=deltag, num_gp=num_gp, \
                                 xmax=self._xmax, num_nodes=self._num_nodes, \
                                 source=self._source, velocities=vel1, \
                                 deltat=deltat, tmax=tmax, stddev=0)

            column = (seismo2.array - seismo1.array)/(2*dvel)
            
            jacobian.append(column)
            
        return numpy.array(jacobian).T 
            
    
    def _calc_adjusted_data(self, estimate):
        """
        Calculate the adjusted data vector based on the current estimate
        """        
        
        deltat = self._seismogram.deltat
        
        offset = self._seismogram.offset
        
        deltag = self._seismogram.deltag
        
        num_gp = self._seismogram.num_geophones
        
        tmax = len(self._seismogram)*deltat
        
        # Build the velocity array to match the model space discretization
        velocities = numpy.zeros(self._num_nodes)
        
        node_dx = self._xmax/(self._num_nodes - 1)
        
        model_dx = self._xmax/(self._nx)
        
        nodes_per_cell = int(model_dx/node_dx) + 1
        
        for i in xrange(self._nparams):
            
            index = i*nodes_per_cell
            
            velocities[index:index + nodes_per_cell] = float(estimate[i])*\
                                                    numpy.ones(nodes_per_cell)

        seismo = Seismogram()
        
        seismo._log.setLevel(logging.WARNING)
        
        seismo.synthetic_1d(offset=offset, deltag=deltag, num_gp=num_gp, \
                             xmax=self._xmax, num_nodes=self._num_nodes, \
                             source=self._source, velocities=velocities, \
                             deltat=deltat, tmax=tmax, stddev=0)
        
        return seismo.array
        
        
    def _build_first_deriv(self):
        """
        Compute the first derivative matrix of the model parameters.
        """        
        
        start = time.clock()
        
        # The number of derivatives there will be
        deriv_num = (self._nx - 1)
        
        first_deriv = numpy.zeros((deriv_num, self._nparams))
                
        # Derivatives in the x direction   
        for i in range(self._nx - 1):                
            
            first_deriv[i][i] = -1
            
            first_deriv[i][i + 1] = 1
        
        end = time.clock()
        self._log.info("Building first derivative matrix: %d x %d  (%g s)" \
                      % (deriv_num, self._nparams, end - start))
        
        return first_deriv
            
            
    def _get_data_array(self):
        """
        Return the data in a Numpy array so that the algorithm can access it
        in a general way
        """        
        
        return self._seismogram.array
                           
            
    def _get_data_cov(self):
        """
        Return the data covariance in a 2D Numpy array so that the algorithm can
        access it in a general way
        """        
        
        return self._seismogram.cov
    
        
    def plot_adjustment(self, title="Adjustment", exaggerate=5000):
        """
        Plot the data seismograms and the adjusted ones.
        """
        
        # Calculate the adjusted data        
        deltat = self._seismogram.deltat
        
        offset = self._seismogram.offset
        
        deltag = self._seismogram.deltag
        
        num_gp = self._seismogram.num_geophones
        
        tmax = len(self._seismogram)*deltat
        
        # Build the velocity array to match the model space discretization
        velocities = numpy.empty(self._num_nodes)
        
        node_dx = self._xmax/(self._num_nodes - 1)
        
        model_dx = self._xmax/(self._nx)
        
        nodes_per_cell = int(model_dx/node_dx) + 1
        
        for i in xrange(self._nparams):
            
            index = i*nodes_per_cell
            
            velocities[index:index + nodes_per_cell] = self.mean[i]*\
                                                    numpy.ones(nodes_per_cell)

        seismo = Seismogram()
        
        seismo._log.setLevel(logging.WARNING)
        
        seismo.synthetic_1d(offset=offset, deltag=deltag, num_gp=num_gp, \
                             xmax=self._xmax, num_nodes=self._num_nodes, \
                             source=self._source, velocities=velocities, \
                             deltat=deltat, tmax=tmax, stddev=0)
        
        pylab.figure()
        
        pylab.title(title)
        
        times = seismo.times
        
        n = seismo.num_geophones
                
        for i in xrange(n):
            
            amplitudes = self._seismogram.get_seismogram(i)
            
            position = self._seismogram.offset + self._seismogram.deltag*i
            
            x = position + exaggerate*amplitudes
            
            pylab.plot(x, times, '-k') 
            
            adjusted = seismo.get_seismogram(i)
            
            adjusted_position = seismo.offset + seismo.deltag*i
            
            adjusted_x = adjusted_position + exaggerate*adjusted
            
            pylab.plot(adjusted_x, times, '-r') 
            
        pylab.xlabel("Offset (m)")
        
        pylab.ylabel("Time (s)")
        
        pylab.ylim(times.max(), 0)
        
        
    def map_goal(self, lower, upper, delta, damping=0, smoothness=0, \
                 sharpness=0):
        
        
        if self._first_deriv == None:
            
            self._first_deriv = self._build_first_deriv()
        
        p2s = numpy.arange(lower[1], upper[1] + delta[1], delta[1])
        
        p1s = numpy.arange(lower[0], upper[0] + delta[0], delta[0])
        
        rms = numpy.zeros((len(p2s), len(p1s)))
        
        goal_tk0 = numpy.zeros((len(p2s), len(p1s)))
        
        goal_tk1 = numpy.zeros((len(p2s), len(p1s)))
        
        goal_tv = numpy.zeros((len(p2s), len(p1s)))
        
        data = self._seismogram.array
        
        for i in xrange(len(p2s)):
            
            for j in xrange(len(p1s)):
                
                p = numpy.array([p1s[j], p2s[i]])
        
                residuals = data - self._calc_adjusted_data(p)
        
                rms[i][j] = (residuals*residuals).sum()     
                
                if damping:
                
                    goal_tk0[i][j] = damping*((p*p).sum())
                
                if smoothness or sharpness:
                    
                    tmp = numpy.dot(self._first_deriv, p)
                    
                    if smoothness:
                    
                        goal_tk1[i][j] = smoothness*((tmp*tmp).sum())
                    
                    if sharpness:
                                                         
                        goal_tv[i][j] = sharpness*(abs(tmp).sum())
                                            
        X, Y = pylab.meshgrid(p1s, p2s)
        
        pylab.savetxt('v1grid.txt', X)
        pylab.savetxt('v2grid.txt', Y)
        
        goal = rms + goal_tk0 + goal_tk1 + goal_tv
        
        pylab.savetxt('rmsgrid.txt', rms)
        pylab.savetxt('tk0grid.txt', goal_tk0)
        pylab.savetxt('tk1grid.txt', goal_tk1)
        pylab.savetxt('tvgrid.txt', goal_tv)
        pylab.savetxt('goalgrid.txt', goal)
        
        pylab.figure()
        
        pylab.title("Goal Function")
        
        CS = pylab.contourf(X, Y, goal, 20)
        
        pylab.colorbar()
        
        pylab.xlabel('v1')
        pylab.ylabel('v2')
        
        pylab.xlim(lower[0], upper[0])
        pylab.ylim(lower[1], upper[1])
        
        pylab.figure()
        
        pylab.title("RMS")
        
        CS = pylab.contourf(X, Y, rms, 20)
        
        pylab.colorbar()
        
        pylab.xlabel('v1')
        pylab.ylabel('v2')
        
        pylab.xlim(lower[0], upper[0])
        pylab.ylim(lower[1], upper[1])
        
        if damping:
            
            pylab.figure()
            
            pylab.title("Tk0 goal function")
            
            CS = pylab.contourf(X, Y, goal_tk0, 20)
            
            pylab.colorbar()
        
            pylab.xlabel('v1')
            pylab.ylabel('v2')
        
            pylab.xlim(lower[0], upper[0])
            pylab.ylim(lower[1], upper[1])
        
        if smoothness:
            
            pylab.figure()
            
            pylab.title("Tk1 goal function")
            
            CS = pylab.contourf(X, Y, goal_tk1, 20)
            
            pylab.colorbar()
        
            pylab.xlabel('v1')
            pylab.ylabel('v2')
        
            pylab.xlim(lower[0], upper[0])
            pylab.ylim(lower[1], upper[1])
                
        if sharpness:
            
            pylab.figure()
            
            pylab.title("TV goal function")
            
            CS = pylab.contourf(X, Y, goal_tv, 20)
            
            pylab.colorbar()
        
            pylab.xlabel('v1')
            pylab.ylabel('v2')
        
            pylab.xlim(lower[0], upper[0])
            pylab.ylim(lower[1], upper[1])