"""
SimpleTom:
    A simplified Cartesian tomography problem. Does not consider reflection or 
    refraction.
"""
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
from fatiando.utils.linearsolver import LinearSolver


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
        
        # Data parameters
        self._src_n = 0
        self._rec_n = 0
        self._sources = []
        self._receivers = []
        
        # The logger for this class
        self._log = logging.getLogger('simpletom')
                  

    def _build_sensibility(self):
        """
        Make the sensibility matrix.
        """
                
        nlines = len(self._data)
        ncolumns = self._mod_sizex*self._mod_sizey
        
        self._log.info("Building sensibility matrix: %d x %d" \
                      % (nlines, ncolumns))
        
        start = time.clock()
        
        sensibility = numpy.zeros((nlines, ncolumns))
        
        for i in range(nlines):
                
            j = 0
            
            for y in range(self._mod_sizey):
                
                for x in range(self._mod_sizex): 
                    
                    sensibility[i][j] = traveltime(\
                                  1, \
                                  x, y, \
                                  x + 1, y + 1, \
                                  self._sources[i][0], self._sources[i][1], \
                                  self._receivers[i][0], self._receivers[i][1])
                    
                    j += 1
                    
        end = time.clock()
        self._log.info("Time it took: %g s" % (end - start))
        
        return sensibility
        
        
    def _build_first_deriv(self):
        """
        Compute the first derivative matrix of the model parameters.
        """
        
        # The number of derivatives there will be
        deriv_num = (self._mod_sizex - 1)*self._mod_sizey + \
                    (self._mod_sizey - 1)*self._mod_sizex
        
        param_num = self._mod_sizex*self._mod_sizey
        
        self._log.info("Building first derivative matrix: %d x %d" \
                      % (deriv_num, param_num))
        start = time.clock()
        
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
        self._log.info("Time it took: %g s" % (end - start))
        
        return first_deriv
        
                       
    def _calc_distances(self, points, lines):
        """
        Calculate the distance from each model element to the closest point or 
        line and put the values in the compact_weights. 
        Also assign the target value for each parameter based on the closest 
        point or line.
        """       
        
        param_num = self._mod_sizex*self._mod_sizey
        
        self._log.info("Calculating distances to points and lines: " + \
                "%d points  %d lines" % (len(points), len(lines)))        
        start = time.clock()
        
        distances = numpy.zeros(param_num)
        
        target_values = numpy.zeros(param_num)
                
        # Find the points with the smallest distance to each cell and set that
        # distance as the weight for the parameter
        for point, value in points:
            
            pnum = 0
            
            for y in range(self._mod_sizey):
                for x in range(self._mod_sizex):
                    
                    deltax = (point[0] - x + 0.5)
                    deltay = (point[1] - y + 0.5)
                    dist_sqr = math.sqrt(deltax**2 + deltay**2)
                    
                    if dist_sqr < distances[pnum] or distances[pnum] == 0:
                        
                        distances[pnum] = dist_sqr
                        
                        target_values[pnum] = value                                    
                                                
                    pnum += 1
        
        end = time.clock()
        self._log.info("Time it took: %g s" % (end - start))
        
        return [distances, target_values]
    
        
    def _build_compact_weights(self, distances, estimate):
        """
        Calculate the weights for the compactness and MMI regularizations.
        'estimate' is the current estimate for the parameters.
        """    
        
        eps = 0.000001
        
        param_num = self._mod_sizex*self._mod_sizey
                
        compact_weights = numpy.zeros((param_num, param_num))
                        
        # Divide the distance by the current estimate
        for i in range(param_num):
            
            compact_weights[i][i] = (distances[i]**2)/ \
                                    (abs(estimate[i]) + eps)
                
        return compact_weights
                                                           
                
    def synthetic_square(self, sizex, sizey, slowness_out=1, slowness_in=2):
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
                
        # Log the model size
        self._log.info("Synthetic model size: %d x %d = %d params" \
                      % (self._mod_sizex, self._mod_sizey, \
                         self._mod_sizex*self._mod_sizey))
        
    
    def synthetic_image(self, image_file, vmin=1, vmax=5):
        """
        Load the synthetic model from an image file. Converts the image to grey
        scale and puts it in the range [vmin,vmax].
        """
        
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
        
        srcs_x = numpy.random.random(src_n)*self._mod_sizex
        srcs_y = numpy.random.random(src_n)*self._mod_sizey
        
        srcs = numpy.array([srcs_x, srcs_y]).T
        
        recs_r = numpy.random.normal(0.48*self._mod_sizex, \
                                     0.02*self._mod_sizex, rec_n)
        recs_theta = numpy.random.random(rec_n)*2*numpy.pi
        
        recs_x = 0.5*self._mod_sizex + recs_r*numpy.cos(recs_theta)
        recs_y = 0.5*self._mod_sizey + recs_r*numpy.sin(recs_theta)
        
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
                        
                        data[l] += traveltime(\
                                           self._model[i][j], \
                                           j, i, \
                                           j + 1, i + 1, \
                                           src[0], src[1], rec[00.01], rec[1])
                        
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
        
            
    def plot_data(self, cmap=pylab.cm.Greys, outdir=None):
        """
        Plot the original model, sources, receivers, ray paths, and histogram of
        the travel times.
        """
        # Plot the model with the sources and receivers
        ########################################################################
        pylab.figure()
        pylab.axis('scaled')
        pylab.title('True Model with Sources and Receivers')
        
        pylab.pcolor(self._model, cmap=cmap, \
                     vmin=numpy.min(self._model), vmax=numpy.max(self._model))
        
        cb = pylab.colorbar(orientation='vertical')
        cb.set_label("Slowness")
        
        pylab.plot(self._sources.T[0], self._sources.T[1], 'r*', ms=9, \
                   label='Source')
        pylab.plot(self._receivers.T[0], self._receivers.T[1], 'b^', ms=7, \
                   label='Receiver')
        
        pylab.legend(numpoints=1, prop={'size':7})
                 
        pylab.xlim(0, self._mod_sizex)
        pylab.ylim(0, self._mod_sizey)   
        
        # Plot the raypaths
        ########################################################################
        pylab.figure()
        pylab.axis('scaled')    
        pylab.title('Raypaths')    
        
        for i in range(len(self._data)):
            pylab.plot([self._sources[i][0], self._receivers[i][0]], \
                       [self._sources[i][1], self._receivers[i][1]], 'k-')      
        
        pylab.plot(self._sources.T[0], self._sources.T[1], 'r*', ms=9, \
                   label='Source')
        pylab.plot(self._receivers.T[0], self._receivers.T[1], 'b^', ms=7, \
                   label='Receiver')
        
        pylab.legend(numpoints=1, prop={'size':7})
                 
        pylab.xlim(0, self._mod_sizex)
        pylab.ylim(0, self._mod_sizey)
        
        # Plot a histogram of the travel times    
        ########################################################################
        pylab.figure()
        pylab.title("Travel times")
        
        pylab.hist(self._data, bins=len(self._data)/8, facecolor='gray')
        
        pylab.xlabel("Travel time")
        pylab.ylabel("Count")
        
        
    def plot_result(self, cmap=pylab.cm.Greys, outdir=None):
        """
        Plot the inversion result (mean of all the estimates), standard 
        deviation, and histogram of the residuals.        
        """
        
        # Make the results into grids so that they can be plotted
        result = numpy.resize(self._mean, (self._mod_sizey, self._mod_sizex))
        stddev = numpy.resize(self._stddev, (self._mod_sizey, self._mod_sizex))
        
        pylab.figure()
        pylab.title("Residuals")
        
        pylab.hist(self._residuals, bins=len(self._residuals)/8, \
                   facecolor='gray')
        
        pylab.xlabel("Residuals")
        pylab.ylabel("Count")
        
        pylab.figure()
        pylab.axis('scaled')        
        pylab.title('Inversion Result')    
        
#        pylab.pcolor(result, cmap=cmap, \
#                     vmin=numpy.min(self._model), vmax=numpy.max(self._model))
        pylab.pcolor(result, cmap=cmap)
        
        cb = pylab.colorbar(orientation='vertical')
        cb.set_label("Slowness")                   
        
        pylab.xlim(0, self._mod_sizex)
        pylab.ylim(0, self._mod_sizey)
        
        pylab.figure()
        pylab.axis('scaled')
        pylab.title('Result Standard Deviation')    
        
        pylab.pcolor(stddev, cmap=pylab.cm.jet)
        
        cb = pylab.colorbar(orientation='vertical')
        cb.set_label("Standard Deviation")  
        
        pylab.xlim(0, self._mod_sizex)
        pylab.ylim(0, self._mod_sizey)
    
    
    def plot_skeleton(self, points, lines):
        """
        Plot the skeleton that was used to estimate the model in invert_compact.
        """
        
        pylab.figure()
        pylab.axis('scaled')
        pylab.title('Model skeleton')    
        
        for point, value in points:
            
            pylab.plot(point[0], point[1], 'o')
            
        for point1, point2, value in lines:
            
            xs = [point1[0], point2[0]]
            ys = [point1[1], point2[1]]
            pylab.plot(xs, ys, '-')
        
        pylab.xlim(0, self._mod_sizex)
        pylab.ylim(0, self._mod_sizey)
    
    
if __name__ == '__main__':
    
    log = logging.getLogger('simpletom')
    
    stom = SimpleTom()    
    
    log.info("* Generating synthetic model")
#    stom.synthetic_square(sizex=30, sizey=30, slowness_out=1, slowness_in=5)
    image = "/home/leo/src/fatiando/examples/simpletom/2src.jpg"
    stom.synthetic_image(image, vmin=1, vmax=8)
    
    log.info("* Shooting rays")
    apriori_std = stom.shoot_rays(src_n=30, rec_n=15, stddev=0.05)  
    
    log.info("* Inverting Tikhonov")
    stom.solve(damping=1, smoothness=3, apriori_var=apriori_std**2, \
               contam_times=20)    
    stom.plot_result()
    
    log.info("* Compacting")
    points = [((8, 23), 8), ((20, 8), 8)]
    lines = []
    stom.compact(points, lines, compact=1, apriori_var=apriori_std**2, \
                contam_times=1, max_iterations=30)
    stom.plot_result()
    
    stom.plot_data()
    stom.plot_skeleton(points, lines)
    pylab.show()
    
