"""
Classes for loading and generating gravity data, such as full tensor data,
vertical gravity and geoid.
   
NOTE: for now only the tensor gravity is being implemented
"""

import logging
import time
from PIL import Image

import pylab
import numpy
import scipy

from fatiando.data import GeoData
from fatiando.utils import points, contaminate
from fatiando.directmodels.gravity import prism
import fatiando


logger = logging.getLogger('gravitydata')       
logger.setLevel(logging.DEBUG)
logger.addHandler(fatiando.default_log_handler)


class VerticalGravity(GeoData):
    """
    Loads, holds and simulates vertical gravity data.
    """
    
    def __init__(self):
        
        GeoData.__init__(self)
        
        # Matrix holding the full data set (location, data, stddev)
        self._data = None
        
        self._log = logging.getLogger('gravitydata.VerticalGravity')
        
        
    def __len__(self):
        
        return len(self._data[3])

        
    def load(self, fname):
        """
        Load the vertical gravity data from file 'fname'.
        Position uncertainties are ignored.
        File must be formated as:
        
            # This is a comment
            # x  y  z  value  stddev
            x1 y1 z1 value1 stddev1
            x2 y2 z2 value2 stddev2
            x3 y3 z3 value3 stddev3
            ...            
            xN yN zN valueN stddevN
            
        Coordinate system is x->north, y->east, and z->down. Coordinates should 
        be in SI and vertical gravity value in mGal.  
        Comments can be put into the file using a # at the start of the comment.
        """
        
        data = pylab.loadtxt(fname, dtype='float', comments='#', unpack=False)
        
        # Need to transpose because the data is in columns (second index) and
        # I don't want that
        self._data = data.T
        
        if len(self._data) != 5:
            
            raise IOError, "Wrong number of columns in file '%s'." % (fname) + \
                           " Has %d, should have exactly 5." % (len(self._data))
                                           
        self._log.info("Loaded %d vertical gravity values from file '%s'" \
                       % (len(self._data[3]), fname))
        
        
    def dump(self, fname):
        """
        Dump the data to file 'fname' in the format:
        
            # This is a comment
            # x  y  z  value  stddev
            x1 y1 z1 value1 stddev1
            x2 y2 z2 value2 stddev2
            x3 y3 z3 value3 stddev3
            ...            
            xN yN zN valueN stddevN
            
        Coordinate system is x->north, y->east, and z->down. Coordinates should 
        be in SI and tensor component value in mGal.  
        Comments can be put into the file using a # at the start of the comment.
        """
                       
        pylab.savetxt(fname, self._data.T, fmt='%f', delimiter=' ')
     
    
        
    def _toarray(self):
        """
        Convert the vertical gravity values to a numpy array.
        """
        
        return numpy.array(self._data[3])
    
    
    array = property(_toarray)    
    
    
    def _get_cov(self):
        """
        Convert the standard deviations to a covariance matrix.
        """
        
        return numpy.diag(self._data[4])**2
    
    
    cov = property(_get_cov)    
    
    
    def togrid(self, nrows, ncolumns):
        """
        Assuming the data is in a regular grid, return a 2D numpy array with the
        data. 
        
        Parameters:
        
            nrows: number of rows in the original grid
            
            ncolumns: number of columns in the original grid
        """
        
        return numpy.reshape(self._data[3], (nrows, ncolumns))
    
    
    def synthetic_interface_from_image(self, image_file, dens, \
                    cell_dx, cell_dy, vmin, vmax, ref_height,\
                    gz, gx1, gx2, gy1, gy2, gnx, gny, stddev):
        """
        Generate synthetic data on a regular grid using a prism model of a 3D 
        interface relief loaded from an image file.
        The reference system is z->down, x->north, y->east
        
        Parameters:
        
            image_file: an image file with the model of type supported by PIL
            
            dens: density contrast between the prisms and the medium
            
            cell_dx, cell_dy: the prism dimensions in the x and y directions
            
            vmin, vmax: reference maximum top and bottom, respectively, of the 
                        prisms. Used to translate the color scale into z 
                        coordinates.
                        
            ref_height: the z coordinate of the reference surface. Heights above
                        this will be tops of prisms and bellow will be bottoms.
                        
            gz: z coordinate of the data grid
            
            gx1, gx2, gy1, gy2: data grid boundaries in the x and y dimensions
            
            gnx, gny: number of points in the data grid in the x and y 
                      dimensions
                      
            stddev: percentage of the maximum data value that will be used as 
                    standard deviation for the errors contaminating the data
                      
        Returns:
        
            2D array with the height of the prisms above and bellow the 
            reference surface
        """
                
        self._log.info("Loading model from image file '%s'" % (image_file))
        
        image = Image.open(image_file)
        
        imagearray = scipy.misc.fromimage(image, flatten=True)
        
        # Invert the color scale
        relief = numpy.max(imagearray) - imagearray
        
        # Normalize
        relief = relief/numpy.max(imagearray)
        
        # Put it in the interval [vmin,vmax]
        relief = relief*(vmax - vmin) + vmin
        
        # Convert the model to a list so that I can reverse it (otherwise the
        # image will be upside down)
        relief = relief.tolist()        
        relief.reverse()        
        relief = numpy.array(relief)
        
        sizey, sizex = relief.shape
        
        # The tops of the prisms are the relief and the bottoms the reference
        # surface.
        tops = relief - ref_height
        
        xs = numpy.arange(0, cell_dx*sizex, cell_dx, dtype='float')
        
        ys = numpy.arange(0, cell_dy*sizey, cell_dy, dtype='float')
        
        # Make the grid points
        gcellx = (gx2 - gx1)/(gnx - 1)
        
        gcelly = (gy2 - gy1)/(gny - 1)                
        
        gxs = numpy.arange(gx1, gx2 + gcellx, gcellx, dtype='float')
        
        gys = numpy.arange(gy1, gy2 + gcelly, gcelly, dtype='float')
        
        values = numpy.zeros((gnx, gny))
        
        start = time.clock()
                    
        # Iterate over the model and then the grid
        for l in xrange(len(ys)):
            
            for m in xrange(len(xs)):
                
                for i in xrange(gny):
                    
                    for j in xrange(gnx):
                        
                        values[i][j] += prism.gz(dens, xs[m], xs[m] + cell_dx, \
                                                 ys[l], ys[l] + cell_dy, \
                                                 tops[l][m], ref_height, \
                                                 gxs[j], gys[i], gz)
        
        end = time.clock()                
        self._log.info("Calculate vertical gravity of the model (%g s)" \
                  % (end - start))
        
        # Make the grids and flatten them to have a x and y for each point           
        gxs, gys = pylab.meshgrid(gxs, gys)
        
        gxs = gxs.flatten()
        
        gys = gys.flatten()
        
        values = values.flatten()
        
        values, error = contaminate.gaussian(values, stddev, percent=True, \
                                             return_stddev=True)
        
        self._data = [gxs, gys, gz*numpy.ones(len(values)), values, \
                      error*numpy.ones(len(values))]
                       
        return relief          
        