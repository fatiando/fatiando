"""
Seismological data structures such as travel times (Cartesian), surface wave
scatering, etc.
   
NOTE: for now only the cartesian travel times are implemented.
"""

import logging
import time

import pylab
import numpy
import scipy
from PIL import Image

from fatiando.data import GeoData
from fatiando.utils import points, contaminate
from fatiando.directmodels import seismo
import fatiando


logger = logging.getLogger('carttraveltime')       
logger.setLevel(logging.DEBUG)
logger.addHandler(fatiando.default_log_handler)


class CartTravelTime(GeoData):
    """
    Loads, holds and simulates Cartesian travel time data. 
    """
    
    def __init__(self):
        
        GeoData.__init__(self)
        
        # Matrix holding the source and receiver locations, travel times and
        # standard deviation
        self._data = None
                
        self._log = logging.getLogger('carttraveltime')
        
        
    def __len__(self):
        
        return len(self._data[4])

        
    def load(self, fname):
        """
        Load the travel time data from file 'fname'.
        Position uncertainties are ignored.
        File must be formated as:
        
            # This is a comment
            src1x srx1y rec1x rec1y traveltime1 stddev1
            src2x srx2y rec2x rec2y traveltime2 stddev2
            src3x srx3y rec3x rec3y traveltime3 stddev3
            ...            
            srcNx srxNy recNx recNy traveltimeN stddevN
            
        srcix, srciy are the coordinates of the ith source and the same goes
        for the receiver (rec) location.
        Comments can be put into the file using a # at the start of the comment.
        """
        
        data = pylab.loadtxt(fname, dtype='float', comments='#', unpack=False)
        
        # Need to transpose because the data is in columns (second index) and
        # I don't want that
        self._data = data.T
        
    def dump(self, fname):
        """
        Dump the data to file 'fname' in the format:
        
            # This is a comment
            src1x srx1y rec1x rec1y traveltime1 stddev1
            src2x srx2y rec2x rec2y traveltime2 stddev2
            src3x srx3y rec3x rec3y traveltime3 stddev3
            ...            
            srcNx srxNy recNx recNy traveltimeN stddevN
            
        srcix, srciy are the coordinates of the ith source and the same goes
        for the receiver (rec) location.
        Comments can be put into the file using a # at the start of the comment.
        """
                       
        pylab.savetxt(fname, self._data.T, fmt='%f', delimiter=' ')
        
    
    def source(self, i):
        """
        Return a point object with the ith source location.
        To access the coordinates x and y use the respective properties in the
        point object.
        
        Example:
        
            data = CartTravelTime()
            data.load('myfile.txt')
            print data.source(1).x, data.source(1).y
        """
        
        return points.Cart2DPoint(self._data[0][i], self._data[1][i])
    

    def receiver(self, i):
        """
        Return a point object with the ith receiver location.
        To access the coordinates x and y use the respective properties in the
        point object.
        
        Example:
        
            data = CartTravelTime()
            data.load('myfile.txt')
            print data.receiver(1).x, data.receiver(1).y
        """
        
        return points.Cart2DPoint(self._data[2][i], self._data[3][i])
    
        
    def _toarray(self):
        """
        Convert the travel times to a numpy array.
        """
        
        return numpy.array(self._data[4])
    
    
    array = property(_toarray)    
    
    
    def _get_cov(self):
        """
        Convert the standard deviations to a covariance matrix.
        """
        
        return numpy.diag(self._data[5])**2
    
    
    cov = property(_get_cov)    
    
    
    def synthetic_image(self, image_file, src_n, rec_n, type='circ', \
                        dx=1, dy=1, vmin=1, vmax=5, stddev=0.01):
        """
        Create a synthetic model from an image file. Converts the image to grey
        scale and puts it in the range [vmin,vmax]. Then shoots rays to generate
        synthetic travel time data.
        
        Parameters:
        
            - image_file: path to an image file of any type supported by the PIL
        
            - src_n: number of sources
            
            - rec_n: number of receivers
            
            - type: type of array the receiver will be set to. Can be
                    'circ' = random circular array around center of the image
                    'xray' = x-ray array with source and receivers rotating
                             around the center of the figure
                    'rand' = both receivers and sources are randomly distributed
        
            - dx, dy: cell size in the x and y dimensions (used for scale only)
            
            - vmin, vmax: slowness interval to scale the RGB values to
            
            - stddev: percentage of the maximum data value that will be used as 
                      standard deviation for the errors contaminating the data
            
        Returns a numpy 2D array with the synthetic image model loaded. Pass it
        along to print_synthetic if you wanna visualize it.            
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
        
        sizey, sizex = model.shape
                
        # Log the model size
        self._log.info("Image model size: %d x %d = %d params" \
                      % (sizex, sizey, sizex*sizey))
        
        if type == 'circ':
            
            srcs_x, srcs_y, recs_x, recs_y = self._circ_src_rec(src_n, rec_n, \
                                                                sizex, sizey, \
                                                                dx, dy)
        elif type == 'xray':
            pass
        
        else:
            
            srcs_x, srcs_y, recs_x, recs_y = self._rand_src_rec(src_n, rec_n, \
                                                                sizex, sizey, \
                                                                dx, dy)
        
        self._log.info("Generated %d sources and %d receivers" % (src_n, rec_n))
                
        traveltimes, stds = self._shoot_rays(model, dx, dy, srcs_x, srcs_y, \
                                            recs_x, recs_y, stddev)
        
        self._data = numpy.array([srcs_x, srcs_y, recs_x, recs_y, traveltimes, \
                                  stds])
                
        return model
        
                
    def _rand_src_rec(self, src_n, rec_n, sizex, sizey, dx, dy):
        """
        Make some sources and receivers randomly distributed in the model region
        """
        
        srcs_x = []
        srcs_y = []        
        recs_x = []
        recs_y = []
        
        # Generate random sources and receivers
        rand_src_x = numpy.random.random(src_n)*sizex*dx
        rand_src_y = numpy.random.random(src_n)*sizey*dy
        rand_rec_x = numpy.random.random(rec_n)*sizex*dx
        rand_rec_y = numpy.random.random(rec_n)*sizey*dy
        
        # Connect each source with all the receivers
        for i in range(src_n):
            
            src_x = rand_src_x[i]
            src_y = rand_src_y[i]
            
            for j in range(rec_n):
                                
                srcs_x.append(src_x)
                srcs_y.append(src_y)
        
                recs_x.append(rand_rec_x[j])
                recs_y.append(rand_rec_y[j])
                
        return [srcs_x, srcs_y, recs_x, recs_y]
    
        
    def _circ_src_rec(self, src_n, rec_n, sizex, sizey, dx, dy):
        """
        Make some sources randomly distributed in the model region and receivers
        normally distributed in a circle.
        """
        
        minsize = min([sizex*dx, sizey*dy])
        
        srcs_x = []
        srcs_y = []        
        recs_x = []
        recs_y = []
        
        # Generate random sources and receivers
        rand_src_x = numpy.random.random(src_n)*sizex*dx
        rand_src_y = numpy.random.random(src_n)*sizey*dy        
        radius = numpy.random.normal(0.48*minsize, 0.02*minsize, rec_n)        
        angle = numpy.random.random(rec_n)*2*numpy.pi                
        rand_rec_x = 0.5*sizex*dx + radius*numpy.cos(angle)       
        rand_rec_y = 0.5*sizey*dy + radius*numpy.sin(angle)
        
        # Connect each source with all the receivers      
        for i in range(src_n):
            
            src_x = rand_src_x[i]
            src_y = rand_src_y[i]
            
            for j in range(rec_n):
                                
                srcs_x.append(src_x)
                srcs_y.append(src_y)
        
                recs_x.append(rand_rec_x[j])
                recs_y.append(rand_rec_y[j])                
        
        return [srcs_x, srcs_y, recs_x, recs_y]
        

    def _shoot_rays(self, model, dx, dy, srcs_x, srcs_y, recs_x, recs_y, \
                    stddev=0.01):
        """
        Shoots rays through the model.
        """
        
        start = time.clock()
                        
        traveltimes = numpy.zeros(len(srcs_x))        
        
        for l in range(len(traveltimes)):
            
            for i in range(0, len(model)):
                
                for j in range(0, len(model[i])):
                   
                    x1 = j*dx
                    x2 = x1 + dx
                    
                    y1 = i*dy
                    y2 = y1 + dy
                    
                    traveltimes[l] += seismo.simple.traveltime(\
                                       model[i][j], \
                                       x1, y1, x2, y2, \
                                       srcs_x[l], srcs_y[l], \
                                       recs_x[l], recs_y[l])
                                   
        traveltimes, data_stddev = contaminate.gaussian( \
                                                 traveltimes, \
                                                 stddev=stddev, \
                                                 percent=True, \
                                                 return_stddev=True)
        
        data_stddev = data_stddev*numpy.ones(len(traveltimes))
        
        # Log the data attributes
        end = time.clock()
        self._log.info("Rays shot: %d" % (len(traveltimes)))
        self._log.info("Data stddev: %g (%g%s)" \
                       % (data_stddev[0], 100*stddev, '%'))
        self._log.info("Time it took: %g s" % (end - start))
        
        return [traveltimes, data_stddev]
    
    
            
    def plot_synthetic(self, model, dx=1, dy=1, title="Synthetic model", \
                       cmap=pylab.cm.Greys):
        """
        Plot the synthetic model with the sources and receivers.
        
        Parameters:
            
            - model: 2D array with the image model (output of synthetic_image)
            
            - dx, dy: model cell sizes
            
            - title: title of the figure
            
            - cmap: a pylab.cm color map object
            
        Note: to view the image use pylab.show()
        """
        
        sizey, sizex = model.shape
        
        xvalues = numpy.arange(0, (sizex + 1)*dx, dx)
        yvalues = numpy.arange(0, (sizey + 1)*dy, dy)
        
        gridx, gridy = pylab.meshgrid(xvalues, yvalues)
        
        pylab.figure()
        pylab.axis('scaled')
        pylab.title(title)
        
        pylab.pcolor(gridx, gridy, model, cmap=cmap, \
                     vmin=numpy.min(model), vmax=numpy.max(model))
        
        cb = pylab.colorbar(orientation='vertical')
        cb.set_label("Slowness")
        
        pylab.plot(self._data[0], self._data[1], 'r*', ms=9, label='Source')
        
        pylab.plot(self._data[2], self._data[3], 'b^', ms=7, label='Receiver')
        
        pylab.legend(numpoints=1, prop={'size':7})
                 
        pylab.xlim(0, sizex*dx)
        pylab.ylim(0, sizey*dy)   
        
        
    def plot_traveltimes(self, title="Travel times", bins=0):
        """
        Plot a histogram of the travel times.
        
        Parameters:
            
            - title: title of the figure
            
            - bins: number of bins (default to len(data)/8)
            
        Note: to view the image use pylab.show()
        """
        
        if bins == 0:
            
            bins = len(self._data[4])/8
        
        pylab.figure()
        pylab.title(title)
        
        pylab.hist(self._data[4], bins=bins, facecolor='gray')
        
        pylab.xlabel("Travel time")
        pylab.ylabel("Count")
                       
        
    def plot_rays(self, model=None, dx=1, dy=1, title="Ray paths", \
                  cmap=pylab.cm.Greys):
        """
        Plot the ray paths. If the data was generated by a synthetic model, plot
        it beneath the rays. 
        
        Parameters:
            
            - model: 2D array with the synthetic model. If data was loaded from
                     a file instead, leave model=None
            
            - dx, dy: model cell sizes (only valid if a model was given)
            
            - title: title of the figure
            
            - cmap: a pylab.cm color map object
            
        Note: to view the image use pylab.show()
        """
                
        pylab.figure()
        pylab.axis('scaled')    
        pylab.title(title)    
        
        if model != None:
            
            sizey, sizex = model.shape
            
            xvalues = numpy.arange(0, (sizex + 1)*dx, dx)
            yvalues = numpy.arange(0, (sizey + 1)*dy, dy)
            
            gridx, gridy = pylab.meshgrid(xvalues, yvalues)
            
            pylab.pcolor(gridx, gridy, model, cmap=cmap, \
                         vmin=numpy.min(model), vmax=numpy.max(model))
            
            cb = pylab.colorbar(orientation='vertical')
            cb.set_label("Slowness")
        
        for i in range(len(self._data[4])):
            
            pylab.plot([self._data[0][i], self._data[2][i]], \
                       [self._data[1][i], self._data[3][i]], 'k-')     
        
        pylab.plot(self._data[0], self._data[1], 'r*', ms=9, label='Source')
        
        pylab.plot(self._data[2], self._data[3], 'b^', ms=7, label='Receiver')
        
        pylab.legend(numpoints=1, prop={'size':7})
                 
        if model != None:
        
            pylab.xlim(0, sizex*dx)
            pylab.ylim(0, sizey*dy)   
    