"""
SimpleTom:
    A simplified Cartesian tomography problem. Does not consider reflection or 
    refraction.
"""

# TODO:
#    Make a load data function
#    Separate the synthetic model size and the model space discretization
#    Make a Data class that can load the data or make it from a synthetic model
#    


__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 29-Apr-2010'

import sys
import time
import logging
import math

from PIL import Image

import scipy
import pylab
import numpy

from fatiando.directmodels.seismo.simple import traveltime
from fatiando.utils import datamani
from fatiando.math import lu
from fatiando.geoinv.linearsolver import LinearSolver


# Set the default handler to the class logger. 
# By default, logging is set to stderr.
################################################################################ 
stlogger = logging.getLogger('simpletom')
       
stlogger.setLevel(logging.DEBUG)

# Create console handler and set level to debug
console_handler = logging.StreamHandler(strm=sys.stderr)

# Create formatter
formatter = logging.Formatter("simpletom> %(message)s")

# Add formatter to the console handler
console_handler.setFormatter(formatter)

# Set verbose level
console_handler.setLevel(logging.INFO)

# Add the console handler to logger
stlogger.addHandler(console_handler)
################################################################################



class SimpleTom(LinearSolver):
    """
    Solver for the inverse problem of a simple Cartesian tomography with no
    refraction or reflection.
    
    Note:        
        Log messages are printed to stderr by default using the logging module.
        If you want to modify the logging, add handlers to the 'simpletom'
        logger.
        Ex:
            mylog = logging.getLogger('simpletom')
            mylog.addHandler(myhandler)
    """
    
    def __init__(self):
        """
        Initialize the parameters of the inversion.
        """
        
        LinearSolver.__init__(self)
        
        # Synthetic model parameters
        self._model = [[]]
        self._mod_sizex = 0
        self._mod_sizey = 0
        self._dx = 0
        self._dy = 0
        
        # Data parameters
        self._sources = []
        self._receivers = []
        
        # The logger for this class
        self._log = logging.getLogger('simpletom')
                  

    def _build_sensibility(self):
        """
        Make the sensibility matrix.
        """
        
        start = time.clock()
                        
        nlines = len(self._data)
        ncolumns = self._mod_sizex*self._mod_sizey
                
        sensibility = numpy.zeros((nlines, ncolumns))
        
        for l in range(nlines):
                
            p = 0
            
            for i in range(self._mod_sizey):
                
                for j in range(self._mod_sizex):
                    
                    x1 = j*self._dx
                    x2 = x1 + self._dx
                        
                    y1 = i*self._dy
                    y2 = y1 + self._dy
                    
                    sensibility[l][p] = traveltime(\
                                  1, \
                                  x1, y1, x2, y2, \
                                  self._sources[l][0], self._sources[l][1], \
                                  self._receivers[l][0], self._receivers[l][1])
                    
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
        deriv_num = (self._mod_sizex - 1)*self._mod_sizey + \
                    (self._mod_sizey - 1)*self._mod_sizex
        
        param_num = self._mod_sizex*self._mod_sizey
        
        first_deriv = numpy.zeros((deriv_num, param_num))
        
        deriv_i = 0
        
        # Derivatives in the x direction        
        param_i = 0
        for i in range(self._mod_sizey):
            
            for j in range(self._mod_sizex - 1):
                
                
                first_deriv[deriv_i][param_i] = 1
                
                first_deriv[deriv_i][param_i + 1] = -1
                
                deriv_i += 1
                
                param_i += 1
            
            param_i += 1
            
        # Derivatives in the y direction        
        param_i = 0
        for i in range(self._mod_sizey - 1):
            
            for j in range(self._mod_sizex):
        
                first_deriv[deriv_i][param_i] = 1
                
                first_deriv[deriv_i][param_i + self._mod_sizex] = -1
                
                deriv_i += 1
                
                param_i += 1
        
        
        end = time.clock()
        self._log.info("Building first derivative matrix: %d x %d  (%g s)" \
                      % (deriv_num, param_num, end - start))
        
        return first_deriv
        
                       
    def _calc_distances(self, points, lines):
        """
        Calculate the distance from each model element to the closest point or 
        line and put the values in the compact_weights. 
        Also assign the target value for each parameter based on the closest 
        point or line.
        """  
        
        start = time.clock()
        
        param_num = self._mod_sizex*self._mod_sizey
        
        distances = numpy.zeros(param_num)
        
        target_values = numpy.zeros(param_num)
                
        # Find the points with the smallest distance to each cell and set that
        # distance as the weight for the parameter
        for point, value in points:
            
            pnum = 0
            
            for i in range(self._mod_sizey):
                for j in range(self._mod_sizex):
                    
                    x = j*self._dx
                    y = i*self._dy
                    
                    deltax = (point[0] - x + 0.5*self._dx)
                    deltay = (point[1] - y + 0.5*self._dy)
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
        
        param_num = self._mod_sizex*self._mod_sizey
                
        compact_weights = numpy.zeros((param_num, param_num))
                        
        # Divide the distance by the current estimate
        for i in range(param_num):
            
            compact_weights[i][i] = (distances[i]**2)/ \
                                    (abs(estimate[i])**2 + eps)
                
        return compact_weights
                                                           
                
    def synthetic_square(self, sizex, sizey, dx=1, dy=1, \
                         slowness_out=1, slowness_in=2):
        """
        Make a square synthetic model with a different slowness region in the
        middle.
        Cuts the model region in cells of 1 x 1.
            
        Parameters:
            
            sizex: size of the model region in the x direction
            
            sizey: size of the model region in the y direction
            
            slowness_out: slowness of the outer region of the model
            
            slowness_in: slowness of the inner region of the model            
        """
        
        self._model = slowness_out*numpy.ones((sizey, sizex))
        
        for i in range(sizey/4, 3*sizey/4 + 1, 1):        
            for j in range(sizex/4, 3*sizex/4 + 1, 1):
                
                self._model[i][j] = slowness_in
        
        self._mod_sizex = sizex
        self._mod_sizey = sizey
        
        self._dx = dx
        self._dy = dy
                
        # Log the model size
        self._log.info("Synthetic model size: %d x %d = %d params" \
                      % (self._mod_sizex, self._mod_sizey, \
                         self._mod_sizex*self._mod_sizey))
        
    
    def synthetic_image(self, image_file, dx=1, dy=1, vmin=1, vmax=5):
        """
        Load the synthetic model from an image file. Converts the image to grey
        scale and puts it in the range [vmin,vmax].
        dx and dy are the cell size in the x and y dimensions.
        """
        self._log.info("Loading model from image file '%s'" % (image_file))
        
        image = Image.open(image_file)
        
        imagearray = scipy.misc.fromimage(image, flatten=True)
        
        # Invert the color scale
        model = numpy.max(imagearray) - imagearray
        
        # Normalize
        model = model/numpy.max(imagearray)
        
        # Put it in the interval [vmin,vmax]
        model = model*(vmax - vmin) + vmin
        
        # Convert the model to a list so that I can reverse it (otherwise the
        # image will be upside down)
        model = model.tolist()        
        model.reverse()
        
        model = numpy.array(model)
        
        self._mod_sizey, self._mod_sizex = model.shape
        
        self._dx = dx
        self._dy = dy
        
        # Log the model size
        self._log.info("Synthetic model size: %d x %d = %d params" \
                      % (self._mod_sizex, self._mod_sizey, \
                         self._mod_sizex*self._mod_sizey))
        
        self._model = model
        
    
    def _random_src_rec(self, src_n, rec_n):
        """
        Make some random sources and receivers. Sources are randomly distributed
        in the model region. Receivers are normally distributed in a circle.
        """
        
        sizex = self._mod_sizex*self._dx
        sizey = self._mod_sizey*self._dy
        
        minsize = min([sizex, sizey])
        
        srcs_x = numpy.random.random(src_n)*sizex
        srcs_y = numpy.random.random(src_n)*sizey
        
        srcs = numpy.array([srcs_x, srcs_y]).T
        
        recs_r = numpy.random.normal(0.48*minsize, 0.02*minsize, rec_n)
        recs_theta = numpy.random.random(rec_n)*2*numpy.pi
        
        recs_x = 0.5*sizex + recs_r*numpy.cos(recs_theta)
        recs_y = 0.5*sizey + recs_r*numpy.sin(recs_theta)
        
        recs = numpy.array([recs_x, recs_y]).T
        
        return [srcs, recs]
        

    def shoot_rays(self, src_n, rec_n, stddev=0.01):
        """
        Creates random sources and receivers and shoots rays through the model.
        Sources are randomly distributed over the model and receivers are 
        distributed in a circle around the center of the model. The cell size of
        the model is assumed to be 1.
        
        Parameters:
        
            src_n: number of sources
            
            rec_n: number of receivers
            
            stddev: percentage of the maximum data value that will be used as 
                    standard deviation for the errors contaminating the data
            
        Note:
            Each source is registered by all receivers.
        
        Returns the standard deviation in data units used to contaminate the 
        data with noise. 
        """
        
        start = time.clock()
        
        # Make the sources and receivers
        srcs, recs = self._random_src_rec(src_n, rec_n)
                
        # Compute the travel times
        data = numpy.zeros(src_n*rec_n)
        self._sources = []
        self._receivers = []
        
        l = 0
        for src in srcs:
            for rec in recs:
                                                       
                for i in range(0, self._mod_sizey):
                    for j in range(0, self._mod_sizex):
                       
                        x1 = j*self._dx
                        x2 = x1 + self._dx
                        
                        y1 = i*self._dy
                        y2 = y1 + self._dy
                        
                        data[l] += traveltime(\
                                           self._model[i][j], \
                                           x1, y1, x2, y2, \
                                           src[0], src[1], rec[0], rec[1])
                            
                self._sources.append(src)
                self._receivers.append(rec)
                
                l += 1
                
        self._data, data_stddev = datamani.contaminate(data, \
                                                      stddev=stddev, \
                                                      percent=True, \
                                                      return_stddev=True)
        
        self._data = numpy.array(self._data)
        self._sources = numpy.array(self._sources)
        self._receivers = numpy.array(self._receivers)
        self._data_cov = data_stddev*numpy.identity(len(self._data))
        
        # Log the data attributes
        end = time.clock()
        self._log.info("Rays shot: %d srcs x %d recs = %d rays" \
                       % (src_n, rec_n, len(self._data)))
        self._log.info("Data stddev: %g (%g%s)" \
                       % (data_stddev, 100*stddev, '%'))
        self._log.info("Time it took: %g s" % (end - start))
        
        return data_stddev    
        
            
    def set_discretization(self, sizex, sizey, dx, dy):
        """
        Set the discretization of the model space to be used in the inversion.
        """
        pass
            
            
    def plot_synthetic(self, title="Synthetic model", cmap=pylab.cm.Greys):
        """
        Plot the synthetic model with the sources and receivers.
        """
        
        xvalues = numpy.arange(0, (self._mod_sizex + 1)*self._dx, self._dx)
        yvalues = numpy.arange(0, (self._mod_sizey + 1)*self._dy, self._dy)
        
        gridx, gridy = pylab.meshgrid(xvalues, yvalues)
        
        pylab.figure()
        pylab.axis('scaled')
        pylab.title(title)
        
        pylab.pcolor(gridx, gridy, self._model, cmap=cmap, \
                     vmin=numpy.min(self._model), vmax=numpy.max(self._model))
        
        cb = pylab.colorbar(orientation='vertical')
        cb.set_label("Slowness")
        
        pylab.plot(self._sources.T[0], self._sources.T[1], 'r*', ms=9, \
                   label='Source')
        pylab.plot(self._receivers.T[0], self._receivers.T[1], 'b^', ms=7, \
                   label='Receiver')
        
        pylab.legend(numpoints=1, prop={'size':7})
                 
        pylab.xlim(0, self._mod_sizex*self._dx)
        pylab.ylim(0, self._mod_sizey*self._dy)   
        
        
    def plot_traveltimes(self, title="Travel times", bins=0):
        """
        Plot a histogram of the travel times with 'bins' number of bins.        
        If bins is zero, the default number of bins (len(data)/8) will be 
        used.
        """
        
        bins = len(self._data)/8
        
        pylab.figure()
        pylab.title("Travel times")
        
        pylab.hist(self._data, bins=bins, facecolor='gray')
        
        pylab.xlabel("Travel time")
        pylab.ylabel("Count")
        
        
    def plot_rays(self, title="Ray paths", cmap=pylab.cm.Greys):
        """
        Plot the ray paths. If the data was generated by a synthetic model, plot
        it beneath the rays. 
        """

        xvalues = numpy.arange(0, (self._mod_sizex + 1)*self._dx, self._dx)
        yvalues = numpy.arange(0, (self._mod_sizey + 1)*self._dy, self._dy)
        
        gridx, gridy = pylab.meshgrid(xvalues, yvalues)
        
        pylab.figure()
        pylab.axis('scaled')    
        pylab.title(title)    
        
        if len(self._model[0]) != 0:
            
            pylab.pcolor(gridx, gridy, self._model, cmap=cmap, \
                         vmin=numpy.min(self._model), \
                         vmax=numpy.max(self._model))
            
            cb = pylab.colorbar(orientation='vertical')
            cb.set_label("Slowness")
        
        for i in range(len(self._data)):
            
            pylab.plot([self._sources[i][0], self._receivers[i][0]], \
                       [self._sources[i][1], self._receivers[i][1]], 'k-')      
        
        pylab.plot(self._sources.T[0], self._sources.T[1], 'r*', ms=9, \
                   label='Source')
        
        pylab.plot(self._receivers.T[0], self._receivers.T[1], 'b^', ms=7, \
                   label='Receiver')
        
        pylab.legend(numpoints=1, prop={'size':7})
                 
        pylab.xlim(0, self._mod_sizex*self._dx)
        pylab.ylim(0, self._mod_sizey*self._dy)
        
        
    def plot_result(self, title='Inversion Result', points=[], lines=[], \
                    cmap=pylab.cm.Greys):
        """
        Plot the inversion result (mean of all the estimates)
        points and lines are the skeleton used in the compact inversion.  
        """
                
        xvalues = numpy.arange(0, (self._mod_sizex + 1)*self._dx, self._dx)
        yvalues = numpy.arange(0, (self._mod_sizey + 1)*self._dy, self._dy)
        
        gridx, gridy = pylab.meshgrid(xvalues, yvalues)
        
        # Make the results into grids so that they can be plotted
        result = numpy.resize(self._mean, (self._mod_sizey, self._mod_sizex))
        
        pylab.figure()
        pylab.axis('scaled')        
        pylab.title(title)    
        
#        pylab.pcolor(gridx, gridy, result, cmap=cmap, \
#                     vmin=numpy.min(self._model), vmax=numpy.max(self._model))
        pylab.pcolor(gridx, gridy, result, cmap=cmap)
        
        cb = pylab.colorbar(orientation='vertical')
        cb.set_label("Slowness")
        
        for point, value in points: 
            
            pylab.plot(point[0], point[1], 'o')
            
        for point1, point2, value in lines:
            
            xs = [point1[0], point2[0]]
            ys = [point1[1], point2[1]]
            pylab.plot(xs, ys, '-')  
        
        pylab.xlim(0, self._mod_sizex*self._dx)
        pylab.ylim(0, self._mod_sizey*self._dy)
        
        
    def plot_residuals(self, title="Residuals", bins=0):
        """
        Plot a histogram of the residuals with 'bins' number of bins.
        If bins is zero, the default number of bins (len(residuals)/8) will be 
        used.
        """
        
        bins = len(self._residuals)/8
    
        pylab.figure()
        pylab.title(title)
        
        pylab.hist(self._residuals, bins=bins, facecolor='gray')
        
        pylab.xlabel("Residuals")
        pylab.ylabel("Count")
        
    
    def plot_std(self, title='Result Standard Deviation', cmap=pylab.cm.Greys):
        """
        Plot the result standard deviation calculated from all the estimates.
        """
                
        xvalues = numpy.arange(0, (self._mod_sizex + 1)*self._dx, self._dx)
        yvalues = numpy.arange(0, (self._mod_sizey + 1)*self._dy, self._dy)
        
        gridx, gridy = pylab.meshgrid(xvalues, yvalues)        
        
        stddev = numpy.resize(self._stddev, (self._mod_sizey, self._mod_sizex))
        
        pylab.figure()
        pylab.axis('scaled')
        pylab.title(title)    
        
        pylab.pcolor(gridx, gridy, stddev, cmap=pylab.cm.jet)
        
        cb = pylab.colorbar(orientation='vertical')
        cb.set_label("Standard Deviation")  
        
        pylab.xlim(0, self._mod_sizex*self._dx)
        pylab.ylim(0, self._mod_sizey*self._dy)
        
        
    def plot_goal(self):
        
        pylab.figure()
        pylab.title("Goal function")
        pylab.xlabel("LM iteration")
        pylab.ylabel("Goal")
        pylab.plot(self._goals, '.-k')
        
        
    
if __name__ == '__main__':
    
    log = logging.getLogger('simpletom')
    
    stom = SimpleTom()    
    
    log.info("*********** Generating synthetic model ***********")
#    stom.synthetic_square(sizex=30, sizey=30, dx=3, dy=3, \
#                          slowness_out=1, slowness_in=5)
#    image = "/home/leo/src/fatiando/examples/simpletom/cabrito.jpg"
    stom.synthetic_image(image, dx=2, dy=2, vmin=1, vmax=8)
    
    log.info("**************** Shooting rays *******************")
    apriori_std = stom.shoot_rays(src_n=100, rec_n=30, stddev=0.01)  
    
    log.info("************** Inverting Tikhonov ****************")
    stom.solve(damping=5, smoothness=3, apriori_var=apriori_std**2, \
               contam_times=20)
    stom.plot_result(title="Tikhonov")
    stom.plot_std(title="Tikhonov Stddev")
    
#    log.info("************** Inverting Tikhonov ****************")
#    stom.solve(damping=10, smoothness=0, apriori_var=apriori_std**2, \
#               contam_times=20)
#    stom.plot_result(cmap=pylab.cm.jet)
    
#    log.info("***************** Sharpening *********************")
#    initial = []
##    stom.clear()
##    initial = 10**(-7)*numpy.ones(900)
#    stom.sharpen(sharpen=0.5, initial_estimate=initial, \
#                 apriori_var=apriori_std**2, \
#                 contam_times=2, max_it=50, max_marq_it=20, \
#                 marq_start=1, marq_step=10)
#    stom.plot_result(title='Total Variation')
#    stom.plot_std(title='Total Variation Stddev')
#    stom.plot_goal()

    
#    log.info("***************** Compacting *********************")
#    stom.clear()
#    points = [((8, 23), 8), ((20, 8), 8)]
#    lines = []
#    stom.compact(points, lines, compact=1, damping=0, smoothness=0, \
#                 apriori_var=apriori_std**2, \
#                 contam_times=0, max_iterations=10)
#    stom.plot_result(points, lines)

    stom.plot_synthetic()
    stom.plot_rays()
    pylab.show()
    
