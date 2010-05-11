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



class SimpleTom():
    """
    Solver for the inverse problem of a simple Cartesian tomography.
    
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
        
        # Synthetic model parameters
        self.model = [[]]
        self.mod_sizex = 0
        self.mod_sizey = 0
        
        # Data parameters
        self.data = []
        self.data_noise_level = 0
        self.data_cov = [[]]
        self.src_n = 0
        self.rec_n = 0
        self.sources = []
        self.receivers = []
        
        # Inversion parameters
        self.sensibility = [[]]
        self.first_deriv = [[]]
        self.compact_weights = [[]]
        self.points = []
        self.lines = []
                
        # Inversion results
        self.estimates = [[]]
        self.means = []
        self.stddevs = []
        self.residuals = []
        
        # The logger for this class
        self.log = logging.getLogger('simpletom')
              
                
    def synthetic_square(self, sizex, sizey, slowness_outer=1, slowness_inner=2):
        """
        Make a square synthetic model with a different slowness region in the
        middle.
        Cuts the model region in cells of 1 x 1.
            
        Parameters:
            
            sizex: size of the model region in the x direction
            
            sizey: size of the model region in the y direction
            
            slowness_outer: slowness of the outer region of the model
            
            slowness_inner: slowness of the inner region of the model            
        """
        
        self.model = slowness_outer*numpy.ones((sizey, sizex))
        
        for i in range(sizey/4, 3*sizey/4 + 1, 1):        
            for j in range(sizex/4, 3*sizex/4 + 1, 1):
                
                self.model[i][j] = slowness_inner
        
        self.mod_sizex = sizex
        self.mod_sizey = sizey
                
        # Log the model size
        self.log.info("Synthetic model size: %d x %d = %d params" \
                      % (self.mod_sizex, self.mod_sizey, \
                         self.mod_sizex*self.mod_sizey))
        
    
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
        # image will be upside down
        model = model.tolist()        
        model.reverse()
        
        model = numpy.array(model)
        
        self.mod_sizey, self.mod_sizex = model.shape
        
        # Log the model size
        self.log.info("Synthetic model size: %d x %d = %d params" \
                      % (self.mod_sizex, self.mod_sizey, \
                         self.mod_sizex*self.mod_sizey))
        
        self.model = model
        
    
    def _random_src_rec(self, src_n, rec_n):
        """
        Make some random sources and receivers. Sources are randomly distributed
        in the model region. Receivers are normally distributed in a circle.
        """
        
        srcs_x = numpy.random.random(src_n)*self.mod_sizex
        srcs_y = numpy.random.random(src_n)*self.mod_sizey
        
        srcs = numpy.array([srcs_x, srcs_y]).T
        
        recs_r = numpy.random.normal(0.48*self.mod_sizex, 0.02*self.mod_sizex, \
                                     rec_n)
        recs_theta = numpy.random.random(rec_n)*2*numpy.pi
        
        recs_x = 0.5*self.mod_sizex + recs_r*numpy.cos(recs_theta)
        recs_y = 0.5*self.mod_sizey + recs_r*numpy.sin(recs_theta)
        
        recs = numpy.array([recs_x, recs_y]).T
        
        return [srcs, recs]
        


    def shoot_rays(self, src_n, rec_n):
        """
        Creates random sources and receivers and shoots rays through the model.
        Sources are randomly distributed over the model and receivers are 
        distributed in a circle around the center of the model. The cell size of
        the model is assumed to be 1.
        """
        
        start = time.clock()
        
        # Make the sources and receivers
        srcs, recs = self._random_src_rec(src_n, rec_n)
        
        # Compute the travel times
        data = numpy.zeros(src_n*rec_n)
        self.sources = []
        self.receivers = []
        
        l = 0
        for src in srcs:
            for rec in recs:
                                                       
                for i in range(0, self.mod_sizey):
                    for j in range(0, self.mod_sizex):
                        
                        data[l] += traveltime(\
                                           self.model[i][j], \
                                           j, i, \
                                           j + 1, i + 1, \
                                           src[0], src[1], rec[0], rec[1])
                        
                self.sources.append(src)
                self.receivers.append(rec)
                
                l += 1
                
        self.data, self.data_noise_level = datamani.contaminate(\
                                                    data, stddev=0.01, \
                                                    percent=True, \
                                                    return_stddev=True)
        
        self.data = numpy.array(self.data)
        self.sources = numpy.array(self.sources)
        self.receivers = numpy.array(self.receivers)
        self.data_cov = self.data_noise_level*numpy.identity(len(self.data))
        
        # Log the data attributes
        end = time.clock()
        self.log.info("Rays shot: %d srcs x %d recs = %d rays" \
                      % (src_n, rec_n, len(self.data)))
        self.log.info("Data stddev: %g (1%s)" % (self.data_noise_level, '%'))
        self.log.info("Time it took: %g s" % (end - start))
        

    def _build_sensibility(self):
        """
        Make the sensibility matrix.
        """
                
        nlines = len(self.data)
        ncolumns = self.mod_sizex*self.mod_sizey
        
        self.log.info("Building sensibility matrix: %d x %d" \
                      % (nlines, ncolumns))
        
        start = time.clock()
        
        self.sensibility = numpy.zeros((nlines, ncolumns))
        
        for i in range(nlines):
                
            j = 0
            
            for y in range(self.mod_sizey):
                
                for x in range(self.mod_sizex): 
                    
                    self.sensibility[i][j] = traveltime(\
                                  1, \
                                  x, y, \
                                  x + 1, y + 1, \
                                  self.sources[i][0], self.sources[i][1], \
                                  self.receivers[i][0], self.receivers[i][1])
                    
                    j += 1
                    
        end = time.clock()
        self.log.info("Time it took: %g s" % (end - start))
        
        
    def _calculate_first_derivs(self):
        """
        Compute the first derivative matrix of the model parameters.
        """
        
        # The number of derivatives there will be
        deriv_num = (self.mod_sizex - 1)*self.mod_sizey + \
                    (self.mod_sizey - 1)*self.mod_sizex
        
        param_num = self.mod_sizex*self.mod_sizey
        
        self.log.info("Building first derivative matrix: %d x %d" \
                      % (deriv_num, param_num))
        start = time.clock()
        
        self.first_deriv = numpy.zeros((deriv_num, param_num))
        
        deriv_i = 0
        
        # Derivatives in the x direction        
        param_i = 0
        for i in range(self.mod_sizey):
            
            for j in range(self.mod_sizex - 1):
                
                
                self.first_deriv[deriv_i][param_i] = 1
                
                self.first_deriv[deriv_i][param_i + 1] = -1
                
                deriv_i += 1
                
                param_i += 1
            
            param_i += 1
            
        # Derivatives in the y direction        
        param_i = 0
        for i in range(self.mod_sizey - 1):
            
            for j in range(self.mod_sizex):
        
                self.first_deriv[deriv_i][param_i] = 1
                
                self.first_deriv[deriv_i][param_i + self.mod_sizex] = -1
                
                deriv_i += 1
                
                param_i += 1
        
        
        end = time.clock()
        self.log.info("Time it took: %g s" % (end - start))
                       
                       
    def _calculate_distances_targets(self):
        """
        Calculate the distance from each cell to the closest point or line and
        put the values in the compact_weights. Also assign the target value for
        each parameter based on the closes point or line.
        """       
        
        param_num = self.mod_sizex*self.mod_sizey
        
        self.log.info("Calculating distances to points and lines: " + \
                "%d points  %d lines" % (len(self.points), len(self.lines)))        
        start = time.clock()
        
        distances = numpy.zeros(param_num)
        
        target_values = numpy.zeros(param_num)
                
        # Find the points with the smallest distance to each cell and set that
        # distance as the weight for the parameter
        for point, value in self.points:
            
            pnum = 0
            
            for y in range(self.mod_sizey):
                for x in range(self.mod_sizex):
                    
                    deltax = (point[0] - x + 0.5)
                    deltay = (point[1] - y + 0.5)
                    dist_sqr = math.sqrt(deltax**2 + deltay**2)
                    
                    if dist_sqr < distances[pnum] or distances[pnum] == 0:
                        
                        distances[pnum] = dist_sqr
                        
                        target_values[pnum] = value                                    
                                                
                    pnum += 1
        
        end = time.clock()
        self.log.info("Time it took: %g s" % (end - start))
        
        return [distances, target_values]
    
        
    def _calculate_compact_weights(self, distances, estimate):
        """
        Calculate the weights for the compactness and mmi regularizations.
        'estimate' is the current estimate for the parameters.
        See docstring for 'invert' for more details.
        """
        
        eps = 0.000001
        
        param_num = self.mod_sizex*self.mod_sizey
        
        self.log.info("Building compactness weight matrix: %d x %d" \
                      % (param_num, param_num))        
        start = time.clock()
        
        self.compact_weights = numpy.zeros((param_num, param_num))
                        
        # Divide the distance by the current estimate
        for i in range(param_num):
            
            self.compact_weights[i][i] = (distances[i]**2)/ \
                                         (abs(estimate[i]) + eps)
        
        end = time.clock()
        self.log.info("Time it took: %g s" % (end - start))
        
        
    def _solve_overdet(self, initial_estimate, param_weights, contam):
        """
        Solve the overdetermined inverse problem given the parameter weights
        and that the sensibility matrix has already been calculated. Solves
        'contam' times for different noise realizations in the data and returns
        the estimates in a list.
        """
        
        self.log.info("Solving overdetermined system:")
            
        # The data weight matrix (inverse of the data covariance)
        data_cov_lu, data_cov_permut = lu.decomp(self.data_cov.tolist())
        Wd = self.data_noise_level* \
             numpy.array(lu.inv(data_cov_lu, data_cov_permut))
        
        # Build the normal equations
        self.log.info("  Building normal equations (N)")
        N = numpy.dot(numpy.dot(self.sensibility.T, Wd), \
                      self.sensibility) + param_weights
                            
        # Do the LU decomposition of N
        start_minor = time.clock()
        self.log.info("  LU decomposition on N")
        
        N_lu, N_permut = lu.decomp(N.tolist())
        
        end_minor = time.clock()
        self.log.info("  Time it took: %g s" % (end_minor - start_minor))
        
        # The initial estimate of the data
        self.log.info("  Computing initial estimate of data")
        initial_data = numpy.dot(self.sensibility, initial_estimate) 
                            
        # Auxiliary vector
        y = numpy.dot(self.sensibility.T, self.data - initial_data)
        
        # Solve Np = y
        start_minor = time.clock()
        self.log.info("  Solving the linear system")
        
        estimate_corr = lu.solve(N_lu, N_permut, y.tolist())
        
        end_minor = time.clock()
        self.log.info("  Time it took: %g s" % (end_minor - start_minor))
                    
        estimates = [initial_estimate + estimate_corr]
        
        # Contaminate the data 'contam' times
        start_minor = time.clock()
        self.log.info("  Contaminating the data")     
               
        for i in range(1, contam+1):
            
            contam_data = datamani.contaminate(self.data, \
                                            stddev=self.data_noise_level, \
                                            percent=False, \
                                            return_stddev=False)
            
            # Aux vector
            y = numpy.dot(self.sensibility.T, contam_data - initial_data)
            
            # Solve Np = y
            estimate_corr = lu.solve(N_lu, N_permut, y.tolist())
            
            estimates.append(initial_estimate + estimate_corr)
            
        end_minor = time.clock()
        self.log.info("  Time it took: %g s" % (end_minor - start_minor))
         
        return estimates
    
                
    def _solve_underdet(self, initial_estimate, param_weights, contam):   
        """
        Solve the underdetermined inverse problem given the parameter weights
        and that the sensibility matrix has already been calculated. Solves
        'contam' times for different noise realizations in the data and returns
        the estimates in a list.
        """     
        
        self.log.info("Solving underdetermined system:")

        # Invert the parameter weight matrix
        self.log.info("  Calculating inverse of parameter weights (Wp)")
        start_minor = time.clock()
        
#            Wp_lu, Wp_permut = lu.decomp(Wp.tolist())
#            Wp_inv = numpy.array(lu.inv(Wp_lu, Wp_permut))
        Wp_inv = numpy.linalg.inv(param_weights)
                    
        end_minor = time.clock()
        self.log.info("  Time it took: %g s" % (end_minor - start_minor))
                      
        # The inverse of the data weight matrix is the normalized covariance
        Wd_inv = (1./self.data_noise_level)*self.data_cov
        
        # Build the normal equations
        self.log.info("  Building normal equations (N)")
        N = numpy.dot(numpy.dot(self.sensibility, Wp_inv), \
                      self.sensibility.T) + Wd_inv
                      
        # Do the LU decomposition of N
#            start_minor = time.clock()
#            self.log.info("  LU decomposition on N")
#            
#            N_lu, N_permut = lu.decomp(N.tolist())
#            
#            end_minor = time.clock()
#            self.log.info("  Time it took: %g s" % (end_minor - start_minor))

        # Calculate N inverse
        start_minor = time.clock()
        self.log.info("  Calculating N inverse")
        
#            N_inv = lu.inv(N_lu, N_permut)
        N_inv = numpy.linalg.inv(N)
        
        end_minor = time.clock()
        self.log.info("  Time it took: %g s" % (end_minor - start_minor))
        
        # Calculate the pseudo-inverse
        self.log.info("  Calculating the pseudo-inverse")
        H = numpy.dot(numpy.dot(Wp_inv, self.sensibility.T), N_inv)
        
        # The initial estimate of the data
        self.log.info("  Computing initial estimate of data")
        initial_data = numpy.dot(self.sensibility, initial_estimate) 

        estimate_corr = numpy.dot(H, self.data - initial_data)

        estimates = [initial_estimate + estimate_corr]
        
        # Contaminate the data 'contam' times
        start_minor = time.clock()
        self.log.info("  Contaminating the data")     
               
        for i in range(1, contam+1):
            
            contam_data = datamani.contaminate(self.data, \
                                            stddev=self.data_noise_level, \
                                            percent=False, \
                                            return_stddev=False)
            
            estimate_corr = numpy.dot(H, contam_data - initial_data)
            
            estimates.append(initial_estimate + estimate_corr)
            
        end_minor = time.clock()
        self.log.info("  Time it took: %g s" % (end_minor - start_minor))
        
        return estimates
            
    def set_skeleton(self, points, lines):
        """
        Set the geometric elements that compose the skeleton of the solution
        in a compact inversion.
        
        'points' is a list of points around which to enforce the compactness of
        the solution. Each point should be a tuple such as: ((x,y,z),value). The
        z coordinate should be omitted in 2 dimensional problems. 'value' is the 
        reference value for the physical property (density, velocity, etc).
        
        'lines' is a list of lines around which to enforce the minimum moment of
        inertia. As in points, each line should be a tuple such as: 
        ((x1,y1,z1),(x2,y2,z2),value) where the (x,y,z) sets are 2 points that
        describe the line. 'z' should be omitted for 2 dimensional problems.        
        """
        
        self.points = points
        self.lines = lines
        
    
    def invert_compact(self, damping=0, smoothness=0, compact=0, contam=10):
        """
        Perform the linear inversion with the Tikhonov regularizations:
        
            * Ridge Regression (Damped);
            
            * Smoothness;
            
            * Compactness + Minimum Moment of Inertia (MMI);
            
        The respective arguments to the function are the regularization 
        parameters. 
        Compactness and MMI have been put together into one weight
        matrix. The two can be separated by sending only points or only lines
        to concentrate the mass around.
        To set the points and lines for the inversion, see method 'set_skeleton'
        
        contam tells how many times to contaminate the data with noise and 
        perform the inversion again. The result will be a collection of a 
        'contam' number of estimates on which some statistics can be performed.
        """        
        
        max_it = 100
        
        self.log.info("Compactness regularization parameters:")
        self.log.info("  * damping = %g" % (damping))
        self.log.info("  * smoothness = %g" % (smoothness))
        self.log.info("  * compactness = %g" % (compact))
        start = time.clock()
        
        self._build_sensibility()
        
        self._calculate_first_derivs()
        
        ndata, nparams = self.sensibility.shape      
                
        # The pure Tikhonov parameter weight matrix
        Wp_tk = damping*numpy.identity(nparams) + \
                smoothness*numpy.dot(self.first_deriv.T, self.first_deriv)
                    
        Wp = Wp_tk + compact*numpy.identity(nparams)  
        
        initial_estimate = numpy.zeros(nparams)
        
        self.stddevs = numpy.zeros(nparams)
        
        self.log.info("Calculating initial estimate")
        
        # Solve the appropriate kind of problem for an estimate
        if nparams < ndata:
            
            self.means = self._solve_overdet(initial_estimate, \
                                             Wp, \
                                             contam=0)[0]
        
        else:
            
            self.means = self._solve_underdet(initial_estimate, \
                                              Wp, \
                                              contam=0)[0]
                        
            
        distances, targets = self._calculate_distances_targets()
        
        self.log.info("Starting iterations for compactness")        
        for iteration in range(max_it):
            
            self.log.info("iteration %d:" % (iteration))            
            
            self._calculate_compact_weights(distances, self.means)
            
            # Check if any parameter was off limits in the previous estimate
            for i in range(nparams):
                
                if self.means[i] > 1.05*targets[i]:
                    
                    self.means[i] = targets[i]
                    
                    self.compact_weights[i][i] = 10**6
                    
                elif self.means[i] < 0:
                    
                    self.means[i] = 0
                    
                    self.compact_weights[i][i] = 10**6
            
            
            Wp = Wp_tk + compact*self.compact_weights
                        
            # Solve the appropriate kind of problem for an estimate
            if nparams < ndata:
                
                self.estimates = self._solve_overdet(self.means, Wp, contam=contam)
            
            else:
                
                self.estimates = self._solve_underdet(self.means, Wp, contam=contam)
                
            # Compute model statistics
            start_minor = time.clock()
            self.log.info("  Computing model statistics")     
                        
            self.estimates = numpy.array(self.estimates)
            l = 0
            for param in self.estimates.T:
                
                self.means[l] = param.mean()
                self.stddevs[l] = param.std()
                
                l += 1
                            
            end_minor = time.clock()
            self.log.info("  Time it took: %g s" % (end_minor - start_minor))
                            
            # Stop if the estimates are all bellow the target values
            num = 0
            for i in range(nparams):
                
                if abs(self.means[i]) <= 1.15*abs(targets[i]):
                    
                    num += 1
            
            if num == nparams:
                
                break
            
        # Compute the residuals
        self.log.info("Calculating residuals")     
        self.residuals = self.data - numpy.dot(self.sensibility, self.means)
                        
        end = time.clock()
        self.log.info("Total time for inversion: %g s" % (end - start))
        
        
        
        
         
    def invert_tikho(self, damping=0, smoothness=0, contam=10):
        """
        Perform the linear inversion with the Tikhonov regularizations:
        
            * 0th order: Ridge Regression (Damped);
            
            * 1st order: Smoothness;
            
        The respective arguments to the function are the regularization 
        parameters. 
        
        contam tells how many times to contaminate the data with noise and 
        perform the inversion again. The result will be a collection of a 
        'contam' number of estimates on which some statistics can be performed.
        """
        
        self.log.info("Tikhonov regularization parameters:")
        self.log.info("  * damping = %g" % (damping))
        self.log.info("  * smoothness = %g" % (smoothness))
        start = time.clock()
                
        self._build_sensibility()
        
        ndata, nparams = self.sensibility.shape
        
        self._calculate_first_derivs()
                
        # The parameter weight matrix
        Wp = damping*numpy.identity(nparams) + \
             smoothness*numpy.dot(self.first_deriv.T, self.first_deriv)

        # The initial solution
        initial_estimate = numpy.zeros(nparams)
                
        # OVERDETERMINED
        if ndata >= nparams:
            
            self.estimates = self._solve_overdet(initial_estimate, Wp, \
                                                 contam=contam)
        
        # UNDERDETERMINED
        else:
            
            self.estimates = self._solve_underdet(initial_estimate, Wp, \
                                                  contam=contam)
            
        # Compute model statistics
        start_minor = time.clock()
        self.log.info("Computing model statistics")     
        
        self.means = numpy.zeros(nparams)
        self.stddevs = numpy.zeros(nparams)
        self.estimates = numpy.array(self.estimates)
        l = 0
        for param in self.estimates.T:
            
            self.means[l] = param.mean()
            self.stddevs[l] = param.std()
            
            l += 1
                        
        end_minor = time.clock()
        self.log.info("  Time it took: %g s" % (end_minor - start_minor))
        
        # Compute the residuals
        self.log.info("Calculating residuals")     
        self.residuals = self.data - numpy.dot(self.sensibility, self.means)
                
        end = time.clock()
        self.log.info("Total time for inversion: %g s" % (end - start))
                        
            
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
        
        pylab.pcolor(self.model, cmap=cmap, \
                     vmin=numpy.min(self.model), vmax=numpy.max(self.model))
        
        cb = pylab.colorbar(orientation='vertical')
        cb.set_label("Slowness")
        
        pylab.plot(self.sources.T[0], self.sources.T[1], 'r*', ms=9, \
                   label='Source')
        pylab.plot(self.receivers.T[0], self.receivers.T[1], 'b^', ms=7, \
                   label='Receiver')
        
        pylab.legend(numpoints=1, prop={'size':7})
                 
        pylab.xlim(0, self.mod_sizex)
        pylab.ylim(0, self.mod_sizey)   
        
        # Plot the raypaths
        ########################################################################
        pylab.figure()
        pylab.axis('scaled')    
        pylab.title('Raypaths')    
        
        for i in range(len(self.data)):
            pylab.plot([self.sources[i][0], self.receivers[i][0]], \
                       [self.sources[i][1], self.receivers[i][1]], 'k-')      
        
        pylab.plot(self.sources.T[0], self.sources.T[1], 'r*', ms=9, \
                   label='Source')
        pylab.plot(self.receivers.T[0], self.receivers.T[1], 'b^', ms=7, \
                   label='Receiver')
        
        pylab.legend(numpoints=1, prop={'size':7})
                 
        pylab.xlim(0, self.mod_sizex)
        pylab.ylim(0, self.mod_sizey)
        
        # Plot a histogram of the travel times    
        ########################################################################
        pylab.figure()
        pylab.title("Travel times")
        
        pylab.hist(self.data, bins=len(self.data)/8, facecolor='gray')
        
        pylab.xlabel("Travel time")
        pylab.ylabel("Count")
        
        
    def plot_result(self, cmap=pylab.cm.Greys, outdir=None):
        """
        Plot the inversion result (mean of all the estimates), standard 
        deviation, and histogram of the residuals.        
        """
        
        # Make the results into grids so that they can be plotted
        result = numpy.resize(self.means, (self.mod_sizey, self.mod_sizex))
        stddev = numpy.resize(self.stddevs, (self.mod_sizey, self.mod_sizex))
        
        pylab.figure()
        pylab.title("Residuals")
        
        pylab.hist(self.residuals, bins=len(self.residuals)/8, facecolor='gray')
        
        pylab.xlabel("Residuals")
        pylab.ylabel("Count")
        
        pylab.figure()
        pylab.axis('scaled')        
        pylab.title('Inversion Result')    
        
        pylab.pcolor(result, cmap=cmap, \
                     vmin=numpy.min(self.model), vmax=numpy.max(self.model))
#        pylab.pcolor(result, cmap=cmap)
        
        cb = pylab.colorbar(orientation='vertical')
        cb.set_label("Slowness")                   
        
        pylab.xlim(0, self.mod_sizex)
        pylab.ylim(0, self.mod_sizey)
        
        pylab.figure()
        pylab.axis('scaled')
        pylab.title('Result Standard Deviation')    
        
        pylab.pcolor(stddev, cmap=pylab.cm.jet)
        
        cb = pylab.colorbar(orientation='vertical')
        cb.set_label("Standard Deviation")  
        
        pylab.xlim(0, self.mod_sizex)
        pylab.ylim(0, self.mod_sizey)
    
    
    def plot_skeleton(self):
        """
        Plot the skeleton that was used to estimate the model in invert_compact.
        """
        
        pylab.figure()
        pylab.axis('scaled')
        pylab.title('Model skeleton')    
        
        for point, value in self.points:
            
            pylab.plot(point[0], point[1], 'o')
            
        for point1, point2, value in self.lines:
            
            xs = [point1[0], point2[0]]
            ys = [point1[1], point2[1]]
            pylab.plot(xs, ys, '-')
        
        pylab.xlim(0, self.mod_sizex)
        pylab.ylim(0, self.mod_sizey)
    
if __name__ == '__main__':
    
    log = logging.getLogger('simpletom')
    
    stom = SimpleTom()    
    
    log.info("* Generating synthetic model")
#    stom.synthetic_square(sizex=30, sizey=30, slowness_outer=1, slowness_inner=5)
    stom.synthetic_image("/home/leo/src/fatiando/examples/simpletom/2src.jpg", \
                     vmin=1, vmax=8)
    
    log.info("* Shooting rays")
    stom.shoot_rays(src_n=20, rec_n=10)  
    
    log.info("* Inverting")    
    stom.invert_tikho(damping=1, smoothness=5, contam=20)
    stom.plot_result()
    
    stom.set_skeleton(points=[((9,23),8),((20,8),8)], lines=[])
    stom.invert_compact(damping=0, smoothness=0.01, compact=5, contam=0)
    stom.plot_result()
    
    stom.plot_data()
    stom.plot_skeleton()
    pylab.show()
    
