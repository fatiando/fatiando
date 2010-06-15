"""
Classes for loading and generating gravity data, such as full tensor data,
vertical gravity and geoid.
   
NOTE: for now only the tensor gravity is being implemented
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 14-Jun-2010'

import logging
import time
from PIL import Image

import pylab
import numpy
import scipy

from fatiando.data import GeoData
from fatiando.utils import points, contaminate
from fatiando.directmodels.gravity import prism as prism_gravity
import fatiando


logger = logging.getLogger('gravitydata')       
logger.setLevel(logging.DEBUG)
logger.addHandler(fatiando.default_log_handler)


class TensorComponent(GeoData):
    """
    Loads, holds and simulates gravity gradient tensor component data.
    """
    
    def __init__(self, component):
        """
        Parameters:
        
            component: either one of 'xx', 'xy', 'xz', 'yy', 'yz', or 'zz'
        """
    
        if component not in ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']:
            
            raise RuntimeError, "Invalid tensor component '%s'" % (component)
        
        GeoData.__init__(self)
        
        self._component = component
        
        # Matrix holding the full data set (location, data, stddev)
        self._data = None
        
        self._log = logging.getLogger('gravitydata.GravityComponent')
        
        self._calculators = {'xx':prism_gravity.gxx, 'xy':prism_gravity.gxy, \
                             'xz':prism_gravity.gxz, 'yy':prism_gravity.gyy, \
                             'yz':prism_gravity.gyz, 'zz':prism_gravity.gzz}
        
        
    def __len__(self):
        
        return len(self._data[3])

        
    def load(self, fname):
        """
        Load the tensor component data from file 'fname'.
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
        be in SI and tensor components in Eotvos.  
        Comments can be put into the file using a # at the start of the comment.
        """
        
        data = pylab.loadtxt(fname, dtype='float', comments='#', unpack=False)
        
        # Need to transpose because the data is in columns (second index) and
        # I don't want that
        self._data = data.T
        
        if len(self._data) != 5:
            
            raise IOError, "Wrong number of columns in file '%s'." % (fname) + \
                           " Has %d, should have exactly 5." % (len(self._data))
                                           
        self._log.info("Loaded %d values from file '%s'" \
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
        be in SI and tensor components in Eotvos.  
        Comments can be put into the file using a # at the start of the comment.
        """
                       
        pylab.savetxt(fname, self._data.T, fmt='%f', delimiter=' ')
     
    
        
    def _toarray(self):
        """
        Convert the tensor component values to a numpy array.
        """
        
        return numpy.array(self._data[3], dtype='float')
    
    
    array = property(_toarray)    
    
    
    def _get_cov(self):
        """
        Convert the standard deviations to a covariance matrix.
        """
        
        return numpy.diag(self._data[4])**2
    
    
    cov = property(_get_cov)    
    
    
    def _get_std(self):
        """
        Return a Numpy array with the data standard deviation
        """
        
        return numpy.copy(self._data[4])
    
    
    std = property(_get_std)
    
    
    def loc(self, i):
        """
        Return the x, y, z location of the ith data
        """
        
        return [self._data[0][i], self._data[1][i], self._data[2][i]]
    
    
    def togrid(self, nrows, ncolumns):
        """
        Assuming the data is in a regular grid, return a 2D numpy array with the
        data. 
        
        Parameters:
        
            nrows: number of rows in the original grid
            
            ncolumns: number of columns in the original grid
        """
        
        return numpy.reshape(self._data[3], (nrows, ncolumns))
    
    
    def get_xgrid(self, nrows, ncolumns):
        """
        Assuming the data is in a regular grid, return a 2D numpy array with the
        x coordinates of the grid points. 
        
        Parameters:
        
            nrows: number of rows in the original grid
            
            ncolumns: number of columns in the original grid
        """
        
        return numpy.reshape(self._data[0], (nrows, ncolumns))
    
    
    def get_ygrid(self, nrows, ncolumns):
        """
        Assuming the data is in a regular grid, return a 2D numpy array with the
        y coordinates of the grid points. 
        
        Parameters:
        
            nrows: number of rows in the original grid
            
            ncolumns: number of columns in the original grid
        """
        
        return numpy.reshape(self._data[1], (nrows, ncolumns))
    
    
    def get_zgrid(self, nrows, ncolumns):
        """
        Assuming the data is in a regular grid, return a 2D numpy array with the
        z coordinates of the grid points. 
        
        Parameters:
        
            nrows: number of rows in the original grid
            
            ncolumns: number of columns in the original grid
        """
        
        return numpy.reshape(self._data[2], (nrows, ncolumns))
    

    def synthetic_prism(self, prism, X, Y, z, stddev=0.01):
        """
        Create synthetic tensor component data from a prism.
        Coordinate system is x->north, y->east, and z->down.
        All units must be SI. Tensor component is calculated in Eotvos
        
        Parameters:
                    
            prism: instance of fatiando.utils.geometry.Prism class (remember to
                   set the density)
            
            X: 2D array-like with the x coordinates of the data grid
            
            Y: 2D array-like with the y coordinates of the data grid
            
            z: z coordinate of the grid plane
            
            stddev: percentage of the maximum data value that will be used as 
                    standard deviation for the errors contaminating the data.
                    set to False if you don't want contaminated data
            
        Note: to generate X and Y, see pylab.meshgrid
        """
        
        self._log.info("Generating %s component data" % (self._component))
        start = time.clock()

        values = numpy.empty_like(X)

        for i in xrange(X.shape[0]):
            
            for j in xrange(X.shape[1]):

                values[i][j] = self._calculators[self._component](\
                                prism.dens, prism.x1, prism.x2, prism.y1, \
                                prism.y2, prism.z1, prism.z2, \
                                float(X[i][j]), float(Y[i][j]), float(z))
                
        xs = X.flatten()
        
        ys = Y.flatten()
        
        zs = z*numpy.ones_like(xs)

        values = values.flatten()
        
        if stddev:
        
            values, error = contaminate.gaussian(values, stddev, percent=True, \
                                             return_stddev=True)
            
        else:
            
            error = 1
        
        errors = error*numpy.ones_like(values)
        
        self._data = numpy.array([xs, ys, zs, values, errors])
        
        end = time.clock()
        self._log.info("%d %s component values generated with %g error (%g s)" \
                       % (len(values), self._component, error, end - start))
        
        
        
        
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
        be in SI and vertical gravity value in mGal.  
        Comments can be put into the file using a # at the start of the comment.
        """
                       
        pylab.savetxt(fname, self._data.T, fmt='%f', delimiter=' ')
     
    
        
    def _toarray(self):
        """
        Convert the vertical gravity values to a numpy array.
        """
        
        return numpy.array(self._data[3], dtype='float')
    
    
    array = property(_toarray)    
    
    
    def _get_cov(self):
        """
        Convert the standard deviations to a covariance matrix.
        """
        values, error = contaminate.gaussian(values, stddev, percent=True, \
                                             return_stddev=True)
        
        self._data = [gxs, gys, gz*numpy.ones(len(values)), values, \
                      error*numpy.ones(len(values))]
        
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
                        
                        values[i][j] += gravity.prism.gz(\
                                        dens, xs[m], xs[m] + cell_dx, \
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
        