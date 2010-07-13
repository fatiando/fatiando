"""
PGrav:
    3D gravity inversion using right rectangular prisms.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 14-Jun-2010'

import time
import logging
import math

import pylab
import numpy
from enthought.mayavi import mlab
from enthought.tvtk.api import tvtk

import fatiando
from fatiando.geoinv.linearsolver import LinearSolver
from fatiando.geoinv.lmsolver import LMSolver
from fatiando.directmodels.gravity import prism as prism_gravity
from fatiando.utils import contaminate

logger = logging.getLogger('pgrav')       
logger.setLevel(logging.DEBUG)
logger.addHandler(fatiando.default_log_handler)



class PGravNL(LinearSolver):
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
        self._nparams = nx*ny*nz
        
        # The logger for this class
        self._log = logging.getLogger('pgravnl')
        
        self._log.info("Model space discretization: %d cells in x *" % (nx) + \
                       " %d cells in y * %d cells in z = %d parameters" \
                       % (ny, nz, nx*ny*nz))
    
    
    def _build_sensibility(self):
        """
        Make the sensibility matrix.
        """
        
        dx = (self._mod_x2 - self._mod_x1)/self._nx
        dy = (self._mod_y2 - self._mod_y1)/self._ny
        dz = (self._mod_z2 - self._mod_z1)/self._nz        
        
        prism_xs = numpy.arange(self._mod_x1, self._mod_x2, dx, 'float')
        prism_ys = numpy.arange(self._mod_y1, self._mod_y2, dy, 'float')
        prism_zs = numpy.arange(self._mod_z1, self._mod_z2, dz, 'float')
        
        sensibility = []
        
        data_types = [(self._gz, prism_gravity.gz), \
                      (self._gxx, prism_gravity.gxx), \
                      (self._gxy, prism_gravity.gxy), \
                      (self._gxz, prism_gravity.gxz), \
                      (self._gyy, prism_gravity.gyy), \
                      (self._gyz, prism_gravity.gyz), \
                      (self._gzz, prism_gravity.gzz)]
        
        for data, calculator in data_types:
            
            if data != None:
                
                for i in xrange(len(data)):
                    
                    xp, yp, zp = data.loc(i)
                    
                    line = numpy.zeros(self._nparams)
                    
                    j = 0
                                                            
                    for z in prism_zs:
                        
                        for y in prism_ys:
                            
                            for x in prism_xs:
                                                                
                                line[j] = calculator(self._dens, \
                                            x, x + dx, \
                                            y, y + dy, \
                                            z, z + dz, \
                                            xp, yp, zp)
                                
                                j += 1
                            
                    sensibility.append(line)
                    
        sensibility = numpy.array(sensibility)
        
        return sensibility
            
    
    def _calc_adjusted_data(self, estimate):
        """
        Calculate the adjusted data vector based on the current estimate
        """
        
        assert self._jacobian != None, "Jacobian not calculated"
        
        adjusted = numpy.dot(self._sensibility, estimate)
                
        return adjusted
        
        
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
        
        data_array = numpy.array([])
        
        data_types = [self._gz, self._gxx, self._gxy, self._gxz, self._gyy, \
                      self._gyz, self._gzz]
        
        for data in data_types:
            
            if data:
                
                data_array = numpy.append(data_array, data.array)
        
        return data_array
                           
            
    def _get_data_cov(self):
        """
        Return the data covariance in a 2D Numpy array so that the algorithm can
        access it in a general way
        """
        
        std_array = numpy.array([])
        
        data_types = [self._gz, self._gxx, self._gxy, self._gxz, self._gyy, \
                      self._gyz, self._gzz]
        
        for data in data_types:
            
            if data:
                
                std_array = numpy.append(std_array, data.std)
        
        return numpy.diag(std_array**2)
                        
    
    def plot_mean3d(self):
        """
        Plot the mean result in 3D using Mayavi
        """
        
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
        
        fig = mlab.figure()
        fig.scene.background = (0, 0, 0)
        fig.scene.camera.pitch(180)
        fig.scene.camera.roll(180)
        source = mlab.pipeline.add_dataset(grid)
        filter = mlab.pipeline.threshold(source)
        axes = mlab.axes(filter, nb_labels=self._nx+1, \
                         extent=[prism_xs[0], prism_xs[-1], \
                                 prism_ys[0], prism_ys[-1], \
                                 prism_zs[0], prism_zs[-1]])
        surf = mlab.pipeline.surface(axes, vmax=model.max(), vmin=model.min())
        surf.actor.property.edge_visibility = 1
        surf.actor.property.line_width = 1.5
        mlab.colorbar(surf, title="Density [g/cm^3]", orientation='vertical', \
                      nb_labels=10)
        
        
    def plot_std3d(self):
        """
        Plot the standard deviation of the results in 3D using Mayavi
        """
        
        dx = (self._mod_x2 - self._mod_x1)/self._nx
        dy = (self._mod_y2 - self._mod_y1)/self._ny      
        dz = (self._mod_z2 - self._mod_z1)/self._nz
        
        prism_xs = numpy.arange(self._mod_x1, self._mod_x2 + dx, dx, 'float')
        prism_ys = numpy.arange(self._mod_y1, self._mod_y2 + dy, dy, 'float')
        prism_zs = numpy.arange(self._mod_z1, self._mod_z2 + dz, dz, 'float')
        
        std = numpy.reshape(self.std, (self._nz, self._ny, self._nx))*0.001
                        
        grid = tvtk.RectilinearGrid()
        grid.cell_data.scalars = std.ravel()
        grid.cell_data.scalars.name = 'Standard Deviation'
        grid.dimensions = (self._nx + 1, self._ny + 1, self._nz + 1)
        grid.x_coordinates = prism_xs
        grid.y_coordinates = prism_ys
        grid.z_coordinates = prism_zs
        
        fig = mlab.figure()
        fig.scene.background = (0, 0, 0)
        fig.scene.camera.pitch(180)
        fig.scene.camera.roll(180)
        source = mlab.pipeline.add_dataset(grid)
        filter = mlab.pipeline.threshold(source)
        axes = mlab.axes(filter, nb_labels=self._nx+1, \
                         extent=[prism_xs[0], prism_xs[-1], \
                                 prism_ys[0], prism_ys[-1], \
                                 prism_zs[0], prism_zs[-1]])
        surf = mlab.pipeline.surface(axes, vmax=std.max(), vmin=std.min())
        surf.actor.property.edge_visibility = 1
        surf.actor.property.line_width = 1.5
        mlab.colorbar(surf, title="Standard Deviation [g/cm^3]", \
                      orientation='vertical', \
                      nb_labels=10)
                            
            
    def plot_adjustment(self, shape, title="Adjustment", cmap=pylab.cm.jet):
        """
        Plot the original data plus the adjusted data with contour lines.
        """
                
        adjusted = numpy.dot(self._sensibility, estimate)
        
        data_types = [(self._gz, ' $g_{z}'), \
                      (self._gxx, ' $g_{xx}'), \
                      (self._gxy, ' $g_{xy}'), \
                      (self._gxz, ' $g_{xz}'), \
                      (self._gyy, ' $g_{yy}'), \
                      (self._gyz, ' $g_{yz}'), \
                      (self._gzz, ' $g_{zz}')]
        
        for data, subtitle in data_types:
            
            if data != None:
            
                this_adjusted = numpy.reshape(adjusted[:len(data)], shape)
            
                adjusted = adjusted[len(data):]
            
                X = data.get_xgrid(*shape)
                
                Y = data.get_ygrid(*shape)
                
                vmin = min([this_adjusted.min(), data.array.min()])
                vmax = max([this_adjusted.max(), data.array.max()])
            
                pylab.figure()
                pylab.axis('scaled')
                pylab.title(title + subtitle)
                CS = pylab.contour(X, Y, this_adjusted, colors='r', \
                                   label="Adjusted", \
                                   vmin=vmin, vmax=vmax)
                pylab.clabel(CS)
                CS = pylab.contour(X, Y, data.togrid(*X.shape), colors='b', \
                              label="Observed", vmin=vmin, vmax=vmax)
                pylab.clabel(CS)
                
                pylab.xlabel("Y")
                pylab.ylabel("X")
                
                pylab.xlim(Y.min(), Y.max())
                pylab.ylim(X.min(), X.max())
                        


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
        
        data_types = [(self._gz, prism_gravity.gz), \
                      (self._gxx, prism_gravity.gxx), \
                      (self._gxy, prism_gravity.gxy), \
                      (self._gxz, prism_gravity.gxz), \
                      (self._gyy, prism_gravity.gyy), \
                      (self._gyz, prism_gravity.gyz), \
                      (self._gzz, prism_gravity.gzz)]
        
        for data, calculator in data_types:
            
            if data != None:
                
                for i in xrange(len(data)):
                    
                    xp, yp, zp = data.loc(i)
                    
                    line = numpy.zeros(self._nx*self._ny*self._nz)
                    
                    j = 0
                                                            
                    for z in prism_zs:
                        
                        for y in prism_ys:
                            
                            for x in prism_xs:
                                                                
                                line[j] = calculator(1, \
                                            x, x + dx, \
                                            y, y + dy, \
                                            z, z + dz, \
                                            xp, yp, zp)
                                
                                j += 1
                            
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
        
        data_array = numpy.array([])
        
        data_types = [self._gz, self._gxx, self._gxy, self._gxz, self._gyy, \
                      self._gyz, self._gzz]
        
        for data in data_types:
            
            if data:
                
                data_array = numpy.append(data_array, data.array)
        
        return data_array
                           
            
    def _get_data_cov(self):
        """
        Return the data covariance in a 2D Numpy array so that the algorithm can
        access it in a general way
        """        
        
        std_array = numpy.array([])
        
        data_types = [self._gz, self._gxx, self._gxy, self._gxz, self._gyy, \
                      self._gyz, self._gzz]
        
        for data in data_types:
            
            if data:
                
                std_array = numpy.append(std_array, data.std)
        
        return numpy.diag(std_array**2)
    
    
    def depth_weights(self, z0, power, normalize=True):
        """
        Calculate and return the depth weight matrix as in Li and Oldenburg 
        (1996)
        
        Parameters:
        
            z0: reference height
            
            power: decrease rate of the kernel
        """
        
        self._log.info("Building depth weights: z0 = %g   power = %g"\
                        % (z0, power))
                      
        weight = numpy.identity(self._nx*self._ny*self._nz)
        
        dz = (self._mod_z2 - self._mod_z1)/self._nz
        
        depths = numpy.arange(self._mod_z1, self._mod_z2, dz, 'float')
        
        l = 0
        
        for depth in depths:
            
            for j in xrange(self._ny*self._nx):
                                    
                weight[l][l] = 1./(math.sqrt(depth + 0.5*dz + z0)**power)
                
                l += 1
        
        if normalize:
            
            weight = weight/weight.max()
        
        return weight
    
    
    def add_equality(self, x, y, z, value):
        """
        Set an equality constraint for the model cell containing point (x, y, z)        
        """
    
        if self._equality_matrix == None:
            
            D = []
            
        else:
            
            D = self._equality_matrix.tolist()
            
        if self._equality_values == None:
            
            p_ref = []
            
        else:
            
            p_ref = self._equality_values.tolist()
        
        dx = (self._mod_x2 - self._mod_x1)/self._nx
        dy = (self._mod_y2 - self._mod_y1)/self._ny
        dz = (self._mod_z2 - self._mod_z1)/self._nz
        
        prism_xs = numpy.arange(self._mod_x1, self._mod_x2, dx, 'float')
        prism_ys = numpy.arange(self._mod_y1, self._mod_y2, dy, 'float')
        prism_zs = numpy.arange(self._mod_z1, self._mod_z2, dz, 'float')
        
        # This is to know what parameter we're talking about
        l = 0
            
        for prism_z in prism_zs:
            
            for prism_y in prism_ys:
                
                for prism_x in prism_xs:
                    
                    if x >= prism_x and x <= prism_x + dx and \
                       y >= prism_y and y <= prism_y + dy and \
                       z >= prism_z and z <= prism_z + dz:
                        
                        p_ref.append(value)
                        
                        tmp = numpy.zeros(self._nx*self._ny*self._nz)
                        
                        tmp[l] = 1
                        
                        D.append(tmp)
                        
                    l += 1
                    
        self._equality_matrix = numpy.array(D)
        
        self._equality_values = numpy.array(p_ref)
        
        
    def multimodal_prior(self, reg_params, prior_means=None, prior_covs=None, \
              apriori_var=1, contam_times=10, param_weights=None):
        
        self._log.info("a priori variance: %g" % (apriori_var))
        
        total_start = time.clock()
                
        self._estimates = []
        
        if self._sensibility == None:
        
            self._sensibility = self._build_sensibility()
        
        ndata, nparams = self._sensibility.shape
        
        start = time.clock()
        
        Wp = numpy.zeros((nparams, nparams))
        
        ref_params = numpy.zeros(nparams)
        
        if prior_covs != None and prior_means != None:
            
            assert len(prior_covs) == len(prior_means), \
                "Must give same number of prior covariances and means"
            
            for i in xrange(len(prior_covs)):
                
                cov_inv = numpy.linalg.inv(prior_covs[i])
            
                Wp = Wp + reg_params[i]*cov_inv
                
                ref_params = ref_params + \
                             reg_params[i]*numpy.dot(cov_inv, prior_means[i])
        
        end = time.clock()
        self._log.info("Build parameter weight matrix (%g s)" % (end - start))
        
        # Overdetermined
        if nparams <= ndata:
            
            self._log.info("Solving overdetermined problem: %d d x %d p" % \
                           (ndata, nparams))      
              
            # Data weight matrix
            start = time.clock()
            
#            Wd = apriori_var*numpy.linalg.inv(self._get_data_cov())
            Wd = numpy.identity(ndata)
            
            end = time.clock()
            self._log.info("  Build data weight matrix (%g s)" % (end - start))          
              
            # The normal equations
            start = time.clock()
            
            aux = numpy.dot(self._sensibility.T, Wd)
            
            N = numpy.dot(aux, self._sensibility) + Wp
                          
            end = time.clock()
            self._log.info("  Build normal equations matrix (%g s)" \
                           % (end - start))  
            
            # Solve the system for the parameters
            start = time.clock()
            
            y = numpy.dot(aux, self._get_data_array()) + ref_params            
            
            estimate = numpy.linalg.solve(N, y)
            
            end = time.clock()
            self._log.info("  Solve linear system (%g s)" % (end - start))
            
            self._estimates.append(estimate)
            
            start = time.clock()
            
            # Contaminate
            for i in range(contam_times):
                
                contam_data = contaminate.gaussian(\
                                          self._get_data_array(), \
                                          stddev=math.sqrt(apriori_var), \
                                          percent=False, return_stddev=False)
                
                y = numpy.dot(aux, contam_data) + ref_params
                
                estimate = numpy.linalg.solve(N, y)
                
                self._estimates.append(estimate)
                
            end = time.clock()
            self._log.info("  Contaminate data %d times " % (contam_times) + \
                           "with Gaussian noise (%g s)" % (end - start))
                   
#        # Underdetermined
#        else:
#            
#            self._log.info("Solving underdetermined problem: %d d x %d p" % \
#                           (ndata, nparams))            
#            
#            # Inverse of the data weight matrix
#            start = time.clock()
#            
#            Wd_inv = self._get_data_cov()/apriori_var
#            
#            end = time.clock()
#            self._log.info("  Inverse of data weight matrix (%g s)" \
#                            % (end - start))
#                        
#            # The inverse of the parameter weight matrix
#            start = time.clock()
#            
#            Wp_inv = numpy.linalg.inv(Wp)
#            
#            end = time.clock()
#            self._log.info("  Inverse parameter weight matrix (%g s)" \
#                            % (end - start))            
#            
#            # The normal equations
#            start = time.clock()
#            
#            aux = numpy.dot(Wp_inv, self._sensibility.T)
#            
#            N = numpy.dot(self._sensibility, aux) + Wd_inv
#            
#            end = time.clock()
#            self._log.info("  Build normal equations matrix (%g s)" \
#                            % (end - start))
#
#            start = time.clock()
#            
#            y = self._get_data_array()
#            
#            if self._equality_matrix != None:
#                
#                tmp_p_eq = equality*numpy.dot(\
#                            numpy.dot(Wp_inv, self._equality_matrix.T), \
#                            self._equality_values)
#                
#                tmp_y_eq = numpy.dot(self._sensibility, tmp_p_eq)
#                
#                y = y + tmp_y_eq
#                
#            lamb = numpy.linalg.solve(N, y)
#            
#            end = time.clock()
#            self._log.info("  Solve for Lagrange multipliers (%g s)" \
#                           % (end - start))
#            
#            start = time.clock()
#            
#            estimate = numpy.dot(aux, lamb)
#            
#            if self._equality_matrix != None:
#                
#                estimate = estimate + tmp_p_eq
#            
#            self._estimates.append(estimate)
#            
#            end = time.clock()
#            self._log.info("  Calculate the estimate (%g s)" \
#                           % (end - start))
#            
#            start = time.clock()
#            
#            # Contaminate
#            for i in range(contam_times):
#                
#                contam_data = contaminate.gaussian( \
#                                          self._get_data_array(), \
#                                          stddev=math.sqrt(apriori_var), \
#                                          percent=False, return_stddev=False)
#                
#                y = contam_data
#            
#                if self._equality_matrix != None:
#                    
#                    y = y + tmp_y_eq
#                
#                lamb = numpy.linalg.solve(N, y)
#                    
#                estimate = numpy.dot(aux, lamb)
#            
#                if self._equality_matrix != None:
#                
#                    estimate = estimate + tmp_p_eq
#                
#                self._estimates.append(estimate)
#                
#            end = time.clock()
#            self._log.info("  Contaminate data %d times " % (contam_times) + \
#                           "with Gaussian noise (%g s)" % (end - start))
                               
        residuals = self._get_data_array() - numpy.dot(self._sensibility, \
                                                       self.mean)
        
        rms = numpy.dot(residuals.T, residuals)
                
        self._log.info("RMS = %g" % (rms))
            
        total_end = time.clock()
        self._log.info("Total time: %g s" % (total_end - total_start))
        
        
        
        
        
                            
            
    def plot_adjustment(self, shape, title="Adjustment", cmap=pylab.cm.jet):
        """
        Plot the original data plus the adjusted data with contour lines.
        """
        
        adjusted = numpy.dot(self._sensibility, self.mean)
        
        if self._gxx:
            
            gxx = numpy.reshape(adjusted[:len(self._gxx)], shape)
            
            adjusted = adjusted[len(self._gxx):]
            
            X = self._gxx.get_xgrid(*shape)
            
            Y = self._gxx.get_ygrid(*shape)
            
            vmin = min([gxx.min(), self._gxx.array.min()])
            vmax = max([gxx.max(), self._gxx.array.max()])
            
            pylab.figure()
            pylab.axis('scaled')
            pylab.title(title + r" $g_{xx}$")
            CS = pylab.contour(X, Y, gxx, colors='r', label="Adjusted", \
                               vmin=vmin, vmax=vmax)
            pylab.clabel(CS)
            CS = pylab.contour(X, Y, self._gxx.togrid(*X.shape), colors='b', \
                          label="Observed", vmin=vmin, vmax=vmax)
            pylab.clabel(CS)
            
            pylab.xlabel("Y")
            pylab.ylabel("X")
            
            pylab.xlim(Y.min(), Y.max())
            pylab.ylim(X.min(), X.max())
        
        if self._gxy:
            
            gxy = numpy.reshape(adjusted[:len(self._gxy)], shape)
            
            adjusted = adjusted[len(self._gxy):]
            
            X = self._gxy.get_xgrid(*shape)
            
            Y = self._gxy.get_ygrid(*shape)
            
            vmin = min([gxy.min(), self._gxy.array.min()])
            vmax = max([gxy.max(), self._gxy.array.max()])
            
            pylab.figure()
            pylab.axis('scaled')
            pylab.title(title + r" $g_{xy}$")
            CS = pylab.contour(X, Y, gxy, colors='r', label="Adjusted", \
                               vmin=vmin, vmax=vmax)
            pylab.clabel(CS)
            CS = pylab.contour(X, Y, self._gxy.togrid(*X.shape), colors='b', \
                          label="Observed", vmin=vmin, vmax=vmax)
            pylab.clabel(CS)
            
            pylab.xlabel("Y")
            pylab.ylabel("X")
            
            pylab.xlim(Y.min(), Y.max())
            pylab.ylim(X.min(), X.max())
        
        if self._gxz:
            
            gxz = numpy.reshape(adjusted[:len(self._gxz)], shape)
            
            adjusted = adjusted[len(self._gxz):]
            
            X = self._gxz.get_xgrid(*shape)
            
            Y = self._gxz.get_ygrid(*shape)
            
            vmin = min([gxz.min(), self._gxz.array.min()])
            vmax = max([gxz.max(), self._gxz.array.max()])
            
            pylab.figure()
            pylab.axis('scaled')
            pylab.title(title + r" $g_{xz}$")
            CS = pylab.contour(X, Y, gxz, colors='r', label="Adjusted", \
                               vmin=vmin, vmax=vmax)
            pylab.clabel(CS)
            CS = pylab.contour(X, Y, self._gxz.togrid(*X.shape), colors='b', \
                          label="Observed", vmin=vmin, vmax=vmax)
            pylab.clabel(CS)
            
            pylab.xlabel("Y")
            pylab.ylabel("X")
            
            pylab.xlim(Y.min(), Y.max())
            pylab.ylim(X.min(), X.max())
        
        if self._gyy:
            
            gyy = numpy.reshape(adjusted[:len(self._gyy)], shape)
            
            adjusted = adjusted[len(self._gyy):]
            
            X = self._gyy.get_xgrid(*shape)
            
            Y = self._gyy.get_ygrid(*shape)
            
            vmin = min([gyy.min(), self._gyy.array.min()])
            vmax = max([gyy.max(), self._gyy.array.max()])
            
            pylab.figure()
            pylab.axis('scaled')
            pylab.title(title + r" $g_{yy}$")
            CS = pylab.contour(X, Y, gyy, colors='r', label="Adjusted", \
                               vmin=vmin, vmax=vmax)
            pylab.clabel(CS)
            CS = pylab.contour(X, Y, self._gyy.togrid(*X.shape), colors='b', \
                          label="Observed", vmin=vmin, vmax=vmax)
            pylab.clabel(CS)
            
            pylab.xlabel("Y")
            pylab.ylabel("X")
            
            pylab.xlim(Y.min(), Y.max())
            pylab.ylim(X.min(), X.max())
        
        if self._gyz:
            
            gyz = numpy.reshape(adjusted[:len(self._gyz)], shape)
            
            adjusted = adjusted[len(self._gyz):]
            
            X = self._gyz.get_xgrid(*shape)
            
            Y = self._gyz.get_ygrid(*shape)
            
            vmin = min([gyz.min(), self._gyz.array.min()])
            vmax = max([gyz.max(), self._gyz.array.max()])
            
            pylab.figure()
            pylab.axis('scaled')
            pylab.title(title + r" $g_{yz}$")
            CS = pylab.contour(X, Y, gyz, colors='r', label="Adjusted", \
                               vmin=vmin, vmax=vmax)
            pylab.clabel(CS)
            CS = pylab.contour(X, Y, self._gyz.togrid(*X.shape), colors='b', \
                          label="Observed", vmin=vmin, vmax=vmax)
            pylab.clabel(CS)
            
            pylab.xlabel("Y")
            pylab.ylabel("X")
            
            pylab.xlim(Y.min(), Y.max())
            pylab.ylim(X.min(), X.max())
        
        if self._gzz:
            
            gzz = numpy.reshape(adjusted[:len(self._gzz)], shape)
            
            adjusted = adjusted[len(self._gzz):]
            
            X = self._gzz.get_xgrid(*shape)
            
            Y = self._gzz.get_ygrid(*shape)
            
            vmin = min([gzz.min(), self._gzz.array.min()])
            vmax = max([gzz.max(), self._gzz.array.max()])
            
            pylab.figure()
            pylab.axis('scaled')
            pylab.title(title + r" $g_{zz}$")
            CS = pylab.contour(X, Y, gzz, colors='r', label="Adjusted", \
                               vmin=vmin, vmax=vmax)
            pylab.clabel(CS)
            CS = pylab.contour(X, Y, self._gzz.togrid(*X.shape), colors='b', \
                          label="Observed", vmin=vmin, vmax=vmax)
            pylab.clabel(CS)
            
            pylab.xlabel("Y")
            pylab.ylabel("X")
            
            pylab.xlim(Y.min(), Y.max())
            pylab.ylim(X.min(), X.max())
        
        
        
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
                        
    
    def plot_mean3d(self):
        """
        Plot the mean result in 3D using Mayavi
        """
        
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
        
        fig = mlab.figure()
        fig.scene.background = (0, 0, 0)
        fig.scene.camera.pitch(180)
        fig.scene.camera.roll(180)
        source = mlab.pipeline.add_dataset(grid)
        filter = mlab.pipeline.threshold(source)
        axes = mlab.axes(filter, nb_labels=self._nx+1, \
                         extent=[prism_xs[0], prism_xs[-1], \
                                 prism_ys[0], prism_ys[-1], \
                                 prism_zs[0], prism_zs[-1]])
        surf = mlab.pipeline.surface(axes, vmax=model.max(), vmin=model.min())
        surf.actor.property.edge_visibility = 1
        surf.actor.property.line_width = 1.5
        mlab.colorbar(surf, title="Density [g/cm^3]", orientation='vertical', \
                      nb_labels=10)
        
        
    def plot_std3d(self):
        """
        Plot the standard deviation of the results in 3D using Mayavi
        """
        
        dx = (self._mod_x2 - self._mod_x1)/self._nx
        dy = (self._mod_y2 - self._mod_y1)/self._ny      
        dz = (self._mod_z2 - self._mod_z1)/self._nz
        
        prism_xs = numpy.arange(self._mod_x1, self._mod_x2 + dx, dx, 'float')
        prism_ys = numpy.arange(self._mod_y1, self._mod_y2 + dy, dy, 'float')
        prism_zs = numpy.arange(self._mod_z1, self._mod_z2 + dz, dz, 'float')
        
        std = numpy.reshape(self.std, (self._nz, self._ny, self._nx))*0.001
                        
        grid = tvtk.RectilinearGrid()
        grid.cell_data.scalars = std.ravel()
        grid.cell_data.scalars.name = 'Standard Deviation'
        grid.dimensions = (self._nx + 1, self._ny + 1, self._nz + 1)
        grid.x_coordinates = prism_xs
        grid.y_coordinates = prism_ys
        grid.z_coordinates = prism_zs
        
        fig = mlab.figure()
        fig.scene.background = (0, 0, 0)
        fig.scene.camera.pitch(180)
        fig.scene.camera.roll(180)
        source = mlab.pipeline.add_dataset(grid)
        filter = mlab.pipeline.threshold(source)
        axes = mlab.axes(filter, nb_labels=self._nx+1, \
                         extent=[prism_xs[0], prism_xs[-1], \
                                 prism_ys[0], prism_ys[-1], \
                                 prism_zs[0], prism_zs[-1]])
        surf = mlab.pipeline.surface(axes, vmax=std.max(), vmin=std.min())
        surf.actor.property.edge_visibility = 1
        surf.actor.property.line_width = 1.5
        mlab.colorbar(surf, title="Standard Deviation [g/cm^3]", \
                      orientation='vertical', \
                      nb_labels=10)
        
        
    def map_goal(self, true, res, lower, upper, dp1, dp2, \
        damping=0, smoothness=0, curvature=0, equality=0, param_weights=None):
        """
        Map the goal function in the parameter space if there are only 2 
        parameters.
        """
        
        if self._nx*self._ny*self._nz != 2:
            
            raise AttributeError, "Can't do this for %d parameters, only 2" \
                % (self._nx*self._ny*self._nz)
                
        if self._sensibility == None:
            
            self._sensibility = self._build_sensibility()
            
        if self._first_deriv == None:
            
            self._first_deriv = self._build_first_deriv()
            
        p1 = numpy.arange(lower[0], upper[0] + dp1, dp1)
        
        p2 = numpy.arange(lower[1], upper[1] + dp2, dp2)
                
        X, Y = pylab.meshgrid(p1, p2)
        
        np1 = len(p1)
        
        np2 = len(p2)
        
        goal = numpy.zeros((np2, np1))
        
        goal_tk0 = numpy.zeros((np2, np1))
        
        goal_tk1 = numpy.zeros((np2, np1))
        
        goal_tk2 = numpy.zeros((np2, np1))
        
        goal_eq = numpy.zeros((np2, np1))
        
        for i in xrange(np2):
            
            for j in xrange(np1):
                
                p = numpy.array([p1[j], p2[i]])
                
                if damping:
                    
                    if param_weights != None:                    
                    
                        goal_tk0[i][j] += damping*numpy.dot(\
                                            numpy.dot(p.T, param_weights), p)
                
                    else:
                        
                        goal_tk0[i][j] += damping*numpy.dot(p.T, p)
                        
                    goal[i][j] += goal_tk0[i][j]
                        
                if smoothness:
                    
                    tmp = numpy.dot(self._first_deriv, p)
                    
                    if param_weights != None:
                        
                        tmp2 = numpy.dot(numpy.dot(self._first_deriv, param_weights), \
                                         p)                                  
                    
                        goal_tk1[i][j] += smoothness*numpy.dot(tmp2.T, tmp2)
                
                    else:
                        
                        goal_tk1[i][j] += smoothness*numpy.dot(tmp.T, tmp)
                        
                    goal[i][j] += goal_tk1[i][j]
                        
                if curvature:
                    
                    tmp = numpy.dot(numpy.dot(self._first_deriv.T, \
                                              self._first_deriv), p)
                    
                    if param_weights != None:
                        
                        tmp2 = numpy.dot(numpy.dot(p.T, param_weights), \
                                         numpy.dot(self._first_deriv.T, \
                                                   self._first_deriv))                                  
                    
                        goal_tk2[i][j] += curvature*numpy.dot(tmp2, tmp)
                
                    else:
                        
                        goal_tk2[i][j] += curvature*numpy.dot(tmp.T, tmp)
                        
                    goal[i][j] += goal_tk2[i][j]
                        
                if equality:
                    
                    tmp = numpy.dot(self._equality_matrix, p) - \
                            self._equality_values
                    
                    goal_eq[i][j] += equality*numpy.dot(tmp.T, tmp)
                        
                    goal[i][j] += goal_eq[i][j]
                    
                r = self._get_data_array() - numpy.dot(self._sensibility, p)
                
                goal[i][j] += numpy.dot(r.T, r)
                
        pylab.figure()
        
        pylab.axis('scaled')
        
        pylab.title("Total Goal Function")
        
        pylab.pcolor(X, Y, goal, cmap=pylab.cm.jet)
        
        pylab.colorbar()        
        
        CS = pylab.contour(X, Y, goal, 20, colors='k')
        
        pylab.plot(true[0], true[1], 'oc', label="True")
        
        pylab.plot(res[0], res[1], '*k', label="Result")
                   
        pylab.legend(numpoints=1, prop={'size':7})

        pylab.xlabel(r'$p_1$')
        
        pylab.ylabel(r'$p_2$')
        
        pylab.xlim(lower[0], upper[0])
        
        pylab.ylim(lower[1], upper[1])
                
        if damping:            
            
            pylab.figure()
            
            pylab.axis('scaled')
            
            pylab.title("Tikhonov 0")
            
            pylab.pcolor(X, Y, goal_tk0, cmap=pylab.cm.jet)
            
            pylab.colorbar()        
            
            CS = pylab.contour(X, Y, goal_tk0, 10, colors='k')
            
            pylab.plot(true[0], true[1], 'oc', label="True")
            
            pylab.plot(res[0], res[1], '*k', label="Result")
                       
            pylab.legend(numpoints=1, prop={'size':7})
    
            pylab.xlabel(r'$p_1$')
            
            pylab.ylabel(r'$p_2$')
            
            pylab.xlim(lower[0], upper[0])
            
            pylab.ylim(lower[1], upper[1])
                
        if smoothness:            
            
            pylab.figure()
            
            pylab.axis('scaled')
            
            pylab.title("Tikhonov 1")
            
            pylab.pcolor(X, Y, goal_tk1, cmap=pylab.cm.jet)
                      
            pylab.colorbar()        
            
            CS = pylab.contour(X, Y, goal_tk1, 10, colors='k')
            
            pylab.clabel(CS)
            
            pylab.plot(true[0], true[1], 'oc', label="True")
            
            pylab.plot(res[0], res[1], '*k', label="Result")
                       
            pylab.legend(numpoints=1, prop={'size':7})
    
            pylab.xlabel(r'$p_1$')
            
            pylab.ylabel(r'$p_2$')
            
            pylab.xlim(lower[0], upper[0])
            
            pylab.ylim(lower[1], upper[1])
                
        if curvature:            
            
            pylab.figure()
            
            pylab.axis('scaled')
            
            pylab.title("Tikhonov 2")
            
            pylab.pcolor(X, Y, goal_tk2, cmap=pylab.cm.jet)
            
            pylab.colorbar()        
            
            CS = pylab.contour(X, Y, goal_tk2, 10, colors='k')
            
            pylab.plot(true[0], true[1], 'oc', label="True")
            
            pylab.plot(res[0], res[1], '*k', label="Result")
                       
            pylab.legend(numpoints=1, prop={'size':7})
    
            pylab.xlabel(r'$p_1$')
            
            pylab.ylabel(r'$p_2$')
            
            pylab.xlim(lower[0], upper[0])      
            
            pylab.ylim(lower[1], upper[1])      
                
        if equality:            
            
            pylab.figure()
            
            pylab.axis('scaled')
            
            pylab.title("Equality")
            
            pylab.pcolor(X, Y, goal_eq, cmap=pylab.cm.jet)
            
            pylab.colorbar()        
            
            CS = pylab.contour(X, Y, goal_eq, 10, colors='k')
            
            pylab.plot(true[0], true[1], 'oc', label="True")
            
            pylab.plot(res[0], res[1], '*k', label="Result")
                       
            pylab.legend(numpoints=1, prop={'size':7})
    
            pylab.xlabel(r'$p_1$')
            
            pylab.ylabel(r'$p_2$')
            
            pylab.xlim(lower[0], upper[0])
            
            pylab.ylim(lower[1], upper[1])