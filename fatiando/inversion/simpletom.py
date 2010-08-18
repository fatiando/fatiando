"""
SimpleTom:
    A simplified Cartesian tomography problem. Does not consider reflection or 
    refraction.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 29-Apr-2010'


import logging

import pylab
import numpy

from fatiando.directmodels.seismo.simple import traveltime
from fatiando.inversion.gradientsolver import GradientSolver
import fatiando


logger = logging.getLogger('simpletom')       
logger.setLevel(logging.DEBUG)
logger.addHandler(fatiando.default_log_handler)


class SimpleTom(GradientSolver):
    """
    Solver for the inverse problem of a simple Cartesian tomography with no
    refraction or reflection. (Like X-ray tomography)
    
    Constructor parameters:
    
        - traveltimedata: an instance of Cart2DTravelTime found in 
                          fatiando.data.seismo with the travel time data to
                          invert
                           
        - x1, x2, y1, y2: boundaries of the model space
        
        - nx, ny: number of cells into which the model space will be cut up in 
                  each direction
    
    To access the mean and standard deviation of the estimates, use 
    mean and std properties
    
    Note:        
        Log messages are printed to stderr by default using the logging module.
        If you want to modify the logging, add handlers to the 'simpletom'
        logger.
        Ex:
            mylog = logging.getLogger('simpletom')
            mylog.addHandler(myhandler)
            
        Another option is to use the default configuration by calling
        logging.basicConfig() inside your script.
    """
    
    
    def __init__(self, traveltimedata, x1, x2, y1, y2, nx, ny):
        
        GradientSolver.__init__(self)
        
        # Data parameters
        self._data = traveltimedata
        self._ndata = len(traveltimedata)    
        
        # Model space parameters
        self._mod_x1 = x1
        self._mod_x2 = x2
        self._mod_y1 = y1
        self._mod_y2 = y2        
        self._nx = nx
        self._ny = ny
        self._nparams = nx*ny
        
        # Inversion parameters
        self._jacobian = None
        
        # The logger for this class
        self._log = logging.getLogger('simpletom')
        
        self._log.info("Model space discretization: %d cells in x *" % (nx) + \
                       " %d cells in y = %d parameters" % (ny, nx*ny))
                  

    def _build_jacobian(self, estimate):
        """
        Make the Jacobian matrix of the function of the parameters.
        """
        
        if self._jacobian != None:
            
            return self._jacobian
                
        dx = float(self._mod_x2 - self._mod_x1)/ self._nx
        dy = float(self._mod_y2 - self._mod_y1)/ self._ny
                                     
        jacobian = numpy.zeros((self._ndata, self._nparams))
        
        for l in xrange(self._ndata):
                
            p = 0
            
            for y1 in numpy.arange(self._mod_y1, self._mod_y2, dy):
                
                for x1 in numpy.arange(self._mod_x1, self._mod_x2, dx):
                   
                    x2 = x1 + dx
                    
                    y2 = y1 + dy
                    
                    jacobian[l][p] = traveltime(1, x1, y1, x2, y2, \
                            self._data.source(l).x, self._data.source(l).y, \
                            self._data.receiver(l).x, self._data.receiver(l).y)
                    
                    p += 1
        
        self._jacobian = jacobian
        
        return jacobian
    
    
    def _calc_adjusted_data(self, estimate):
        """
        Calculate the adjusted data vector based on the current estimate
        """
        
        if self._jacobian == None:
            
            self._jacobian = self._build_jacobian(estimate)
        
        adjusted = numpy.dot(self._jacobian, estimate)
        
        return adjusted
        
        
    def _build_first_deriv(self):
        """
        Compute the first derivative matrix of the model parameters.
        """
        
        # The number of derivatives there will be
        deriv_num = (self._nx - 1)*self._ny + (self._ny - 1)*self._nx
                
        first_deriv = numpy.zeros((deriv_num, self._nparams))
        
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
            
        
    def plot_mean(self, title='Inversion Result', vmin=None, vmax=None, \
                  cmap=pylab.cm.Greys):
        """
        Plot the mean of all the estimates.
        
        Parameters:
            
            - title: title of the figure
                       
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
        
        pylab.xlim(self._mod_x1, self._mod_x2)
        pylab.ylim(self._mod_y1, self._mod_y2)
                        
    
    def plot_stddev(self, title='Result Standard Deviation', \
                    cmap=pylab.cm.Greys):
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
        
        stddev = numpy.resize(self.stddev, (self._ny, self._nx))
        
        pylab.figure()
        pylab.axis('scaled')
        pylab.title(title)    
        
        pylab.pcolor(gridx, gridy, stddev, cmap=cmap)
        
        cb = pylab.colorbar(orientation='vertical')
        cb.set_label("Standard Deviation")  
                
        pylab.xlim(self._mod_x1, self._mod_x2)
        pylab.ylim(self._mod_y1, self._mod_y2)
