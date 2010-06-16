"""
SimpleTom:
    A simplified Cartesian tomography problem. Does not consider reflection or 
    refraction.
"""

__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 29-Apr-2010'


import time
import logging
import math

import pylab
import numpy

from fatiando.directmodels.seismo.simple import traveltime
from fatiando.utils import contaminate
from fatiando.math import lu
from fatiando.geoinv.linearsolver import LinearSolver
import fatiando


logger = logging.getLogger('simpletom')       
logger.setLevel(logging.DEBUG)
logger.addHandler(fatiando.default_log_handler)


class SimpleTom(LinearSolver):
    """
    Solver for the inverse problem of a simple Cartesian tomography with no
    refraction or reflection.
    
    Parameters:
    
        - traveltimedata: an instance of CartTravelTime found in 
                          fatiando.data.seismo with the travel time data to
                          invert
                           
        - x1, x2, y1, y2: boundaries of the model space
        
        - nx, ny: number of cells into which the model space will be cut up in 
                  each direction
    
    To access the mean and standard deviation of the estimates, use 
    simpletom_instance.mean and simpletom_instance.std
    
    Note:        
        Log messages are printed to stderr by default using the logging module.
        If you want to modify the logging, add handlers to the 'simpletom'
        logger.
        Ex:
            mylog = logging.getLogger('simpletom')
            mylog.addHandler(myhandler)
    """
    
    
    def __init__(self, traveltimedata, x1, x2, y1, y2, nx, ny):
        
        LinearSolver.__init__(self)
        
        self._data = traveltimedata
        
        # Model space parameters
        self._mod_x1 = x1
        self._mod_x2 = x2
        self._mod_y1 = y1
        self._mod_y2 = y2        
        self._nx = nx
        self._ny = ny
        
        # The logger for this class
        self._log = logging.getLogger('simpletom')
        
        self._log.info("Model space discretization: %d cells in x *" % (nx) + \
                       " %d cells in y = %d parameters" % (ny, nx*ny))
                  

    def _build_sensibility(self):
        """
        Make the sensibility matrix.
        """
        
        start = time.clock()
        
        dx = float(self._mod_x2 - self._mod_x1)/ self._nx
        dy = float(self._mod_y2 - self._mod_y1)/ self._ny
                        
        nlines = len(self._data)
        ncolumns = self._nx*self._ny
                
        sensibility = numpy.zeros((nlines, ncolumns))
        
        for l in range(nlines):
                
            p = 0
            
            for y1 in numpy.arange(self._mod_y1, self._mod_y2, dy):
                
                for x1 in numpy.arange(self._mod_x1, self._mod_x2, dx):
                   
                    x2 = x1 + dx
                    
                    y2 = y1 + dy
                    
                    sensibility[l][p] = traveltime(1, x1, y1, x2, y2, \
                            self._data.source(l).x, self._data.source(l).y, \
                            self._data.receiver(l).x, self._data.receiver(l).y)
                    
                    p += 1
                    
        end = time.clock()
        self._log.info("Build sensibility matrix: %d x %d  (%g s)" \
                      % (nlines, ncolumns, end - start))
        
        return sensibility
        
        
    def _build_first_deriv(self):
        """
        Compute the first derivative matrix of the model parameters.
        """
        
        start = time.clock()
        
        # The number of derivatives there will be
        deriv_num = (self._nx - 1)*self._ny + (self._ny - 1)*self._nx
        
        param_num = self._nx*self._ny
        
        first_deriv = numpy.zeros((deriv_num, param_num))
        
        deriv_i = 0
        
        # Derivatives in the x direction        
        param_i = 0
        for i in range(self._ny):
            
            for j in range(self._nx - 1):                
                
                first_deriv[deriv_i][param_i] = 1
                
                first_deriv[deriv_i][param_i + 1] = -1
                
                deriv_i += 1
                
                param_i += 1
            
            param_i += 1
            
        # Derivatives in the y direction        
        param_i = 0
        for i in range(self._ny - 1):
            
            for j in range(self._nx):
        
                first_deriv[deriv_i][param_i] = 1
                
                first_deriv[deriv_i][param_i + self._nx] = -1
                
                deriv_i += 1
                
                param_i += 1        
        
        end = time.clock()
        self._log.info("Building first derivative matrix: %d x %d  (%g s)" \
                      % (deriv_num, param_num, end - start))
        
        return first_deriv
        
        
    def _get_data_array(self):
        """
        Return the data in a Numpy array so that the algorithm can access it
        in a general way
        """        
        
        return self._data.array
                           
            
    def _get_data_cov(self):
        """
        Return the data covariance in a 2D Numpy array so that the algorithm can
        access it in a general way
        """        
        
        return self._data.cov
            
                       
    def _calc_distances(self, points, lines):
        """
        Calculate the distance from each model element to the closest point or 
        line and put the values in the compact_weights. 
        Also assign the target value for each parameter based on the closest 
        point or line.
        """  
        
        start = time.clock()
        
        param_num = self._nx*self._ny
        
        dx = float(self._mod_x2 - self._mod_x1)/ self._nx
        dy = float(self._mod_y2 - self._mod_y1)/ self._ny
        
        distances = numpy.zeros(param_num)
        
        target_values = numpy.zeros(param_num)
                
        # Find the points with the smallest distance to each cell and set that
        # distance as the weight for the parameter
        for point, value in points:
            
            pnum = 0
            
            for y1 in numpy.arange(self._mod_y1, self._mod_y2, dy):
                
                for x1 in numpy.arange(self._mod_x1, self._mod_x2, dx):
                                        
                    deltax = (point[0] - x1 + 0.5*dx)
                    deltay = (point[1] - y1 + 0.5*dy)
                    
                    dist_sqr = math.sqrt(deltax**2 + deltay**2)
                    
                    if dist_sqr < distances[pnum] or distances[pnum] == 0:
                        
                        distances[pnum] = dist_sqr
                        
                        target_values[pnum] = value                                    
                                                
                    pnum += 1
        
        end = time.clock()        
        self._log.info("Calculate distances to points and lines: " + \
                "%d points  %d lines  (%g s)" % (len(points), len(lines), \
                                                 end - start))      
                
        return [distances, target_values]
    
        
    def _build_compact_weights(self, distances, estimate):
        """
        Calculate the weights for the compactness and MMI regularizations.
        'estimate' is the current estimate for the parameters.
        """    
        
        eps = 0.000000001
        
        param_num = self._nx*self._ny
                
        compact_weights = numpy.zeros((param_num, param_num))
                        
        # Divide the distance by the current estimate
        for i in range(param_num):
            
            compact_weights[i][i] = (distances[i]**2)/ \
                                    (abs(estimate[i])**2 + eps)
                
        return compact_weights
            
        
    def plot_mean(self, title='Inversion Result', points=[], lines=[], \
                  vmin=None, vmax=None, cmap=pylab.cm.Greys):
        """
        Plot the mean of all the estimates plus the points and lines of the 
        skeleton used in the compact inversion (optional)  
        
        Parameters:
            
            - title: title of the figure
            
            - points, lines: points and lines used in the compact inversion
           
            - vmin, vmax: mininum and maximum values in the color scale (if not
                          given, the max and min of the result will be used)
            
            - cmap: a pylab.cm color map object
            
        Note: to view the image use pylab.show()
        """
        
        dx = float(self._mod_x2 - self._mod_x1)/self._nx
        dy = float(self._mod_y2 - self._mod_y1)/self._ny
                
        xvalues = numpy.arange(self._mod_x1, self._mod_x2 + dx, dx)
        yvalues = numpy.arange(self._mod_y1, self._mod_y2 + dy, dy)
        
        gridx, gridy = pylab.meshgrid(xvalues, yvalues)
        
        # Make the results into grids so that they can be plotted
        result = numpy.resize(self.mean, (self._ny, self._nx))
        
        pylab.figure()
        pylab.axis('scaled')        
        pylab.title(title)    
        
        if vmin == None or vmax == None:
            
            pylab.pcolor(gridx, gridy, result, cmap=cmap)
        
        else:
            
            pylab.pcolor(gridx, gridy, result, vmin=vmin, vmax=vmax, cmap=cmap)
        
        cb = pylab.colorbar(orientation='vertical')
        cb.set_label("Slowness")
        
        for point, value in points: 
            
            pylab.plot(point[0], point[1], 'o')
            
        for point1, point2, value in lines:
            
            xs = [point1[0], point2[0]]
            ys = [point1[1], point2[1]]
            
            pylab.plot(xs, ys, '-')  
        
        pylab.xlim(self._mod_x1, self._mod_x2)
        pylab.ylim(self._mod_y1, self._mod_y2)
                        
    
    def plot_std(self, title='Result Standard Deviation', cmap=pylab.cm.Greys):
        """
        Plot the result standard deviation calculated from all the estimates. 
        
        Parameters:
            
            - title: title of the figure
                        
            - cmap: a pylab.cm color map object
            
        Note: to view the image use pylab.show()
        """
        
        dx = float(self._mod_x2 - self._mod_x1)/self._nx
        dy = float(self._mod_y2 - self._mod_y1)/self._ny
                
        xvalues = numpy.arange(self._mod_x1, self._mod_x2 + dx, dx)
        yvalues = numpy.arange(self._mod_y1, self._mod_y2 + dy, dy)
        
        gridx, gridy = pylab.meshgrid(xvalues, yvalues)        
        
        stddev = numpy.resize(self.std, (self._ny, self._nx))
        
        pylab.figure()
        pylab.axis('scaled')
        pylab.title(title)    
        
        pylab.pcolor(gridx, gridy, stddev, cmap=cmap)
        
        cb = pylab.colorbar(orientation='vertical')
        cb.set_label("Standard Deviation")  
                
        pylab.xlim(self._mod_x1, self._mod_x2)
        pylab.ylim(self._mod_y1, self._mod_y2)
                   
            
    def plot_goal(self, title="Goal function", scale='log'):
        """
        Plot the goal function versus the iterations of the Levemberg-Marquardt
        algorithm. 
        
        scale is the scale type for the y axis. Can be either 'log' or 'linear'
        """
        
        pylab.figure()
        pylab.title(title)
        
        pylab.xlabel("LM iteration")
        pylab.ylabel("Goal")
        
        pylab.plot(self._goals, '.-k')
        
        if scale == 'log':
            
            ax = pylab.gca()
            
            ax.set_yscale('log')
            
            pylab.draw()
        
        