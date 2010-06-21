"""
PGrav:
    3D gravity inversion using right rectangular prisms.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 14-Jun-2010'

import time
import logging

import pylab
import numpy
from enthought.mayavi import mlab
from enthought.tvtk.api import tvtk

import fatiando
from fatiando.geoinv.linearsolver import LinearSolver
from fatiando.directmodels.gravity import prism as prism_gravity

logger = logging.getLogger('pgrav')       
logger.setLevel(logging.DEBUG)
logger.addHandler(fatiando.default_log_handler)



class PGrav(LinearSolver):
    """
    3D gravity inversion using right rectangular prisms
    """
    
    
    def __init__(self, x1, x2, y1, y2, z1, z2, nx, ny, nz, gz=None, gxx=None, \
                 gxy=None, gxz=None, gyy=None, gyz=None, gzz=None):
        """
        Parameters:
        
            x1, x2, y1, y2, z1, z2: boundaries of the model space
            
            nx, ny, nz: number of prisms into which the model space will be cut
                in the x, y, and z directions
                        
            gz: instance of fatiando.data.gravity.VerticalGravity holding the
                vertical gravity data
                
            gxx, gxy, gxz, gyy, gyz, gzz: instances of 
                fatiando.data.gravity.TensorComponent holding each a respective
                gravity gradient tensor component data
                
        Note: at least of one gz, gxx, gxy, gxz, gyy, gyz, or gzz must be 
        provided            
        """
        
        LinearSolver.__init__(self)
        
        if not (gz or gxx or gxy or gxz or gyy or gyz or gzz):
            
            raise RuntimeError, "Provide at least one of gz, gxx, gxy, gxz," + \
                " gyy, gyz, or gzz. Can't do the inversion without data!"
                
        self._gz = gz
        self._gxx = gxx
        self._gxy = gxy
        self._gxz = gxz
        self._gyy = gyy
        self._gyz = gyz
        self._gzz = gzz
        
        # Model space parameters
        self._mod_x1 = float(x1)
        self._mod_x2 = float(x2)
        self._mod_y1 = float(y1)
        self._mod_y2 = float(y2)  
        self._mod_z1 = float(z1)
        self._mod_z2 = float(z2)        
        self._nx = nx
        self._ny = ny
        self._nz = nz
        
        # The logger for this class
        self._log = logging.getLogger('pgrav')
        
        self._log.info("Model space discretization: %d cells in x *" % (nx) + \
                       " %d cells in y * %d cells in z = %d parameters" \
                       % (ny, nz, nx*ny*nz))
        
        
    def _build_sensibility(self):
        """
        Make the sensibility matrix.
        """
        
        start = time.clock()
        
        dx = (self._mod_x2 - self._mod_x1)/self._nx
        dy = (self._mod_y2 - self._mod_y1)/self._ny      
        dz = (self._mod_z2 - self._mod_z1)/self._nz
        
        prism_xs = numpy.arange(self._mod_x1, self._mod_x2, dx, 'float')
        prism_ys = numpy.arange(self._mod_y1, self._mod_y2, dy, 'float')
        prism_zs = numpy.arange(self._mod_z1, self._mod_z2, dz, 'float')
                
        sensibility = [] 
        
        if self._gz:
            
            for i in xrange(len(self._gz)):
                
                xp, yp, zp = self._gz.loc(i)
                
                line = []
                
                for z in prism_zs:
                    
                    for y in prism_ys:
                        
                        for x in prism_xs:
                            
                            value = prism_gravity.gz(1., x, x + dx, y, y + dy, \
                                        z, z + dz, xp, yp, zp)
                            
                            line.append(value)
            
                sensibility.append(line)
                
        if self._gxx:
            
            for i in xrange(len(self._gxx)):
                
                xp, yp, zp = self._gxx.loc(i)
                
                line = []
                
                for z in prism_zs:
                    
                    for y in prism_ys:
                        
                        for x in prism_xs:
                            
                            value = prism_gravity.gxx(1., x, x + dx, y, y + dy, \
                                        z, z + dz, xp, yp, zp)
                            
                            line.append(value)
            
                sensibility.append(line)
                
        if self._gxy:
            
            for i in xrange(len(self._gxy)):
                
                xp, yp, zp = self._gxy.loc(i)
                
                line = []
                
                for z in prism_zs:
                    
                    for y in prism_ys:
                        
                        for x in prism_xs:
                            
                            value = prism_gravity.gxy(1., x, x + dx, y, y + dy, \
                                        z, z + dz, xp, yp, zp)
                            
                            line.append(value)
            
                sensibility.append(line)
                
        if self._gxz:
            
            for i in xrange(len(self._gxz)):
                
                xp, yp, zp = self._gxz.loc(i)
                
                line = []
                
                for z in prism_zs:
                    
                    for y in prism_ys:
                        
                        for x in prism_xs:
                            
                            value = prism_gravity.gxz(1., x, x + dx, y, y + dy, \
                                        z, z + dz, xp, yp, zp)
                            
                            line.append(value)
            
                sensibility.append(line)
                
        if self._gyy:
            
            for i in xrange(len(self._gyy)):
                
                xp, yp, zp = self._gyy.loc(i)
                
                line = []
                
                for z in prism_zs:
                    
                    for y in prism_ys:
                        
                        for x in prism_xs:
                            
                            value = prism_gravity.gyy(1., x, x + dx, y, y + dy, \
                                        z, z + dz, xp, yp, zp)
                            
                            line.append(value)
            
                sensibility.append(line)
                
        if self._gyz:
            
            for i in xrange(len(self._gyz)):
                
                xp, yp, zp = self._gyz.loc(i)
                
                line = []
                
                for z in prism_zs:
                    
                    for y in prism_ys:
                        
                        for x in prism_xs:
                            
                            value = prism_gravity.gyz(1., x, x + dx, y, y + dy, \
                                        z, z + dz, xp, yp, zp)
                            
                            line.append(value)
            
                sensibility.append(line)
                
        if self._gzz:
            
            for i in xrange(len(self._gzz)):
                
                xp, yp, zp = self._gzz.loc(i)
                
                line = []
                
                for z in prism_zs:
                    
                    for y in prism_ys:
                        
                        for x in prism_xs:
                            
                            value = prism_gravity.gzz(1., x, x + dx, y, y + dy, \
                                        z, z + dz, xp, yp, zp)
                            
                            line.append(value)
            
                sensibility.append(line)
        
        sensibility = numpy.array(sensibility)
        
        end = time.clock()
        self._log.info("Build sensibility matrix: %d x %d  (%g s)" \
                      % (sensibility.shape[0], sensibility.shape[1], \
                         end - start))
        
        return sensibility
    
        
    def _build_first_deriv(self):
        """
        Compute the first derivative matrix of the model parameters.
        """
        
        start = time.clock()
        
        # The number of derivatives there will be
        deriv_num = (self._nx - 1)*self._ny*self._nz + \
                    (self._ny - 1)*self._nx*self._nz + \
                    (self._nz - 1)*self._nx*self._ny
        
        param_num = self._nx*self._ny*self._nz
        
        first_deriv = numpy.zeros((deriv_num, param_num))
        
        deriv_i = 0
        
        # Derivatives in the x direction        
        param_i = 0
        
        for k in xrange(self._nz):
            
            for j in xrange(self._ny):
                
                for i in xrange(self._nx - 1):                
                    
                    first_deriv[deriv_i][param_i] = 1
                    
                    first_deriv[deriv_i][param_i + 1] = -1
                    
                    deriv_i += 1
                    
                    param_i += 1
                
                param_i += 1
            
        # Derivatives in the y direction        
        param_i = 0
        
        for k in xrange(self._nz):
        
            for j in range(self._ny - 1):
                
                for i in range(self._nx):
            
                    first_deriv[deriv_i][param_i] = 1
                    
                    first_deriv[deriv_i][param_i + self._nx] = -1
                    
                    deriv_i += 1
                    
                    param_i += 1
                    
            param_i += self._nx
            
        # Derivatives in the z direction        
        param_i = 0
        
        for k in xrange(self._nz - 1):
        
            for j in range(self._ny):
                
                for i in range(self._nx):
            
                    first_deriv[deriv_i][param_i] = 1
                    
                    first_deriv[deriv_i][param_i + self._nx*self._ny] = -1
                    
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
        
        data = numpy.array([])
        
        if self._gz:
            
            data = numpy.append(data, self._gz.array)
            
        if self._gxx:
            
            data = numpy.append(data, self._gxx.array)
            
        if self._gxy:
            
            data = numpy.append(data, self._gxy.array)
            
        if self._gxz:
            
            data = numpy.append(data, self._gxz.array)
            
        if self._gyy:
            
            data = numpy.append(data, self._gyy.array)
            
        if self._gyz:
            
            data = numpy.append(data, self._gyz.array)
            
        if self._gzz:
            
            data = numpy.append(data, self._gzz.array)
            
        return data                           
                           
            
    def _get_data_cov(self):
        """
        Return the data covariance in a 2D Numpy array so that the algorithm can
        access it in a general way
        """        
        
        stddev = numpy.array([])
        
        if self._gz:
            
            stddev = numpy.append(stddev, self._gz.std)
            
        if self._gxx:
            
            stddev = numpy.append(stddev, self._gxx.std)
            
        if self._gxy:
            
            stddev = numpy.append(stddev, self._gxy.std)
            
        if self._gxz:
            
            stddev = numpy.append(stddev, self._gxz.std)
            
        if self._gyy:
            
            stddev = numpy.append(stddev, self._gyy.std)
            
        if self._gyz:
            
            stddev = numpy.append(stddev, self._gyz.std)
            
        if self._gzz:
            
            stddev = numpy.append(stddev, self._gzz.std)
            
        return numpy.diag(stddev**2)
    
    
    def plot_mean(self):
        """
        Plot the mean solution in layers.
        """
                        
        dx = (self._mod_x2 - self._mod_x1)/self._nx
        dy = (self._mod_y2 - self._mod_y1)/self._ny      
        dz = (self._mod_z2 - self._mod_z1)/self._nz
        
        prism_xs = numpy.arange(self._mod_x1, self._mod_x2 + dx, dx, 'float')
        prism_ys = numpy.arange(self._mod_y1, self._mod_y2 + dy, dy, 'float')
        
        Y, X = pylab.meshgrid(prism_ys, prism_xs)
        
        model = numpy.reshape(self.mean, (self._nz, self._ny, self._nx))*0.001
        
        z = self._mod_z1
        
        for layer in model:
            
            pylab.figure()
            pylab.axis('scaled')
            pylab.title("Result layer z=%g" % (z))
            pylab.pcolor(Y, X, layer.T, vmin=model.min(), vmax=model.max())
            cb = pylab.colorbar()
            cb.set_label("Density [g/cm^3]")
            
            pylab.xlabel("Y")
            pylab.ylabel("X")
            
            pylab.xlim(Y.min(), Y.max())
            pylab.ylim(X.min(), X.max())
            
            z += dz
    
    @mlab.show
    def plot_mean3d(self):
                
        dx = (self._mod_x2 - self._mod_x1)/self._nx
        dy = (self._mod_y2 - self._mod_y1)/self._ny      
        dz = (self._mod_z2 - self._mod_z1)/self._nz
        
        prism_xs = numpy.arange(self._mod_x1, self._mod_x2 + dx, dx, 'float')
        prism_ys = numpy.arange(self._mod_y1, self._mod_y2 + dy, dy, 'float')
        prism_zs = numpy.arange(self._mod_z1, self._mod_z2 + dz, dz, 'float')
        
        model = numpy.reshape(self.mean, (self._nz, self._ny, self._nx))*0.001
                        
        grid = tvtk.RectilinearGrid()
        grid.cell_data.scalars = model.ravel()
        grid.cell_data.scalars.name = 'Density'
        grid.dimensions = (self._nx + 1, self._ny + 1, self._nz + 1)
        grid.x_coordinates = prism_xs
        grid.y_coordinates = prism_ys
        grid.z_coordinates = prism_zs
        
        mlab.pipeline.surface(grid)
                    
    
    def plot_std(self):
        """
        Plot the standard deviation of the model parameters in layers.
        """
        
        dx = (self._mod_x2 - self._mod_x1)/self._nx
        dy = (self._mod_y2 - self._mod_y1)/self._ny      
        dz = (self._mod_z2 - self._mod_z1)/self._nz
        
        prism_xs = numpy.arange(self._mod_x1, self._mod_x2 + dx, dx, 'float')
        prism_ys = numpy.arange(self._mod_y1, self._mod_y2 + dy, dy, 'float')
        
        Y, X = pylab.meshgrid(prism_ys, prism_xs)
        
        stds = numpy.reshape(self.std, (self._nz, self._ny, self._nx))*0.001
        
        z = self._mod_z1
        
        for layer in stds:
            
            pylab.figure()
            pylab.axis('scaled')
            pylab.title("Standard Deviation layer z=%g" % (z))
            
            pylab.pcolor(Y, X, layer.T)
            cb = pylab.colorbar()
            cb.set_label("Standard Deviation [g/cm^3]")
            
            pylab.xlabel("Y")
            pylab.ylabel("X")
            
            pylab.xlim(Y.min(), Y.max())
            pylab.ylim(X.min(), X.max())
            
            z += dz
            
        