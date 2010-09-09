"""
PGrav:
    Gravity inversion for density using right rectangular prisms.
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

logger = logging.getLogger('PGrav')       
logger.setLevel(logging.DEBUG)
logger.addHandler(fatiando.default_log_handler)

logger = logging.getLogger('DepthWeightsCalculator')       
logger.setLevel(logging.DEBUG)
logger.addHandler(fatiando.default_log_handler)

from fatiando.inversion.gradientsolver import GradientSolver
from fatiando.directmodels.gravity import prism as prism_gravity



class DepthWeightsCalculator(GradientSolver):
    """
    Solves for the coefficients of the depth weighing function used in PGrav
    based on the decay of the zz component.
    
    Constructor parameters:
    
        pgrav_solver: instance of PGrav3D for which the weight coefficients will
                      be calculated
                      
        height: height of the observations (positive upward)
    """


    def __init__(self, pgrav_solver, height):
        
        GradientSolver.__init__(self)
       
        self._pgrav_solver = pgrav_solver
        
        self._height = height
                
        self._nparams = 2
        
        self._dz = (pgrav_solver._mod_z2 - pgrav_solver._mod_z1)/ \
                   pgrav_solver._nz
                   
        self._depths = numpy.arange(pgrav_solver._mod_z1, \
                                    pgrav_solver._mod_z2, self._dz, 'float')
        
        self._ndata = len(self._depths)
        
        self._data = None
        
        self._log = logging.getLogger('DepthWeightsCalculator')
        
        
    def _build_jacobian(self, estimate):
        """
        Make the Jacobian matrix of the function of the parameters.
        """
        
        assert estimate != None, "Can't use solve_linear. " + \
            "This is a non-linear inversion!"
        
        jacobian = []
        
        z0 = estimate[0]
        
        power = estimate[1]
                        
        for z in self._depths:
                        
            z0_deriv = -power/((z + 0.5*self._dz + z0)**(power + 1))
            
            power_deriv = -1./((z + 0.5*self._dz + z0)**power)
            
            jacobian.append([z0_deriv, power_deriv])
            
        return numpy.array(jacobian)        
            
    
    def _calc_adjusted_data(self, estimate):
        """
        Calculate the adjusted data vector based on the current estimate
        """
        
        z0 = estimate[0]
        
        power = estimate[1]
        
        adjusted = []
        
        for z in self._depths: 
        
            adjusted.append(1./((z + 0.5*self._dz + z0)**power))
            
        return numpy.array(adjusted)
            
        
    def _build_first_deriv(self):
        """
        Compute the first derivative matrix of the model parameters.
        """
        
        # Raise an exception if the method was raised without being implemented
        raise NotImplementedError, \
            "Only Tikhonov order 0 (damping) is implemented for this solver"
            
            
    def _get_data_array(self):
        """
        Return the data in a Numpy array so that the algorithm can access it
        in a general way
        """        
        
        if self._data != None:
            
            return self._data
        
        data = []
        
        dx = (self._pgrav_solver._mod_x2 - self._pgrav_solver._mod_x1)/ \
             self._pgrav_solver._nx
             
        dy = (self._pgrav_solver._mod_y2 - self._pgrav_solver._mod_y1)/ \
             self._pgrav_solver._ny
                        
        for depth in self._depths:
            
            tmp = prism_gravity.gzz(1., -0.5*dx, 0.5*dx, -0.5*dy, 0.5*dy, \
                            depth, depth + self._dz, 0., 0., -self._height)
                
            data.append(tmp)
        
        self._data = numpy.array(data)
        
        return self._data
                           
            
    def _get_data_cov(self):
        """
        Return the data covariance in a 2D Numpy array so that the algorithm can
        access it in a general way
        """        
        
        return numpy.identity(len(self._depths))
    
    
    def set_equality(self, z0=None, power=None):
        """
        Set an equality constraint for the parameters z0 and/or power.
        """
        
        self._equality_matrix = []
          
        self._equality_values = []
        
        if z0 != None:
            
            self._equality_values.append(z0)
            
            self._equality_matrix.append([1, 0])
            
        if power != None:
                        
            self._equality_values.append(power)
            
            self._equality_matrix.append([0, 1])
            
        self._equality_values = numpy.array(self._equality_values)
        
        self._equality_matrix = numpy.array(self._equality_matrix)
        
    
    def plot_adjustment(self, title="Depth weights adjustment"):
        """
        Plot the depth weights versus the decay of gzz with depth of the model
        cell.
        """
                
        weights = self._calc_adjusted_data(self.mean)
            
        pylab.figure()
        pylab.title(title)
        
        yaxis = self._depths + 0.5*self._dz
        
        pylab.plot(self._get_data_array(), yaxis, '-b', label='$g_{zz}$')
        
        pylab.plot(weights, yaxis, '-r', label='Weights')
        
        pylab.legend(loc='lower right')
        
        pylab.xlabel('Decay')
        
        pylab.ylabel('Depth')
        
        pylab.ylim(yaxis.max(), yaxis.min())
        




class PGrav3D(GradientSolver):
    """
    3D gravity inversion for density using right rectangular prisms.
    
    Constructor parameters:
    
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
    
    Log messages are printed to stderr by default using the logging module.
    If you want to modify the logging, add handlers to the 'simpletom'
    logger.
    Ex:
        mylog = logging.getLogger('simpletom')
        mylog.addHandler(myhandler)
        
    Another option is to use the default configuration by calling
    logging.basicConfig() inside your script.   
    """
    
    
    def __init__(self, x1, x2, y1, y2, z1, z2, nx, ny, nz, gz=None, gxx=None, \
                 gxy=None, gxz=None, gyy=None, gyz=None, gzz=None):
        
        GradientSolver.__init__(self)
        
        if not (gz != None or gxx != None or gxy != None or gxz != None or \
                gyy != None or gyz != None or gzz != None):
            
            raise RuntimeError, "Provide at least one of gz, gxx, gxy, gxz," + \
                " gyy, gyz, or gzz. Can't do the inversion without data!"
                
        self._gz = gz
        self._gxx = gxx
        self._gxy = gxy
        self._gxz = gxz
        self._gyy = gyy
        self._gyz = gyz
        self._gzz = gzz
        
        self._ndata = 0
        
        for data in [gz, gxx, gxy, gxz, gyy, gyz, gzz]:
            
            if data != None:
                
                self._ndata += len(data)
        
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
        
        # Inversion parameters
        self._jacobian = None
        
        # The logger for this class
        self._log = logging.getLogger('PGrav')
        
        self._log.info("Model space discretization: %d cells in x *" % (nx) + \
                       " %d cells in y * %d cells in z = %d parameters" \
                       % (ny, nz, nx*ny*nz))


    def _build_jacobian(self, estimate):
        """
        Make the Jacobian matrix of the function of the parameters.
        """
        
        if self._jacobian != None:
            
            return self._jacobian

        jacobian = []
        
        dx = (self._mod_x2 - self._mod_x1)/self._nx
        dy = (self._mod_y2 - self._mod_y1)/self._ny
        dz = (self._mod_z2 - self._mod_z1)/self._nz
        
        prism_xs = numpy.arange(self._mod_x1, self._mod_x2, dx, 'float')
        prism_ys = numpy.arange(self._mod_y1, self._mod_y2, dy, 'float')
        prism_zs = numpy.arange(self._mod_z1, self._mod_z2, dz, 'float')
                
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
                                                                
                                line[j] = calculator(1, \
                                            x, x + dx, \
                                            y, y + dy, \
                                            z, z + dz, \
                                            xp, yp, zp)
                                
                                j += 1
                            
                    jacobian.append(line)
        
        self._jacobian = numpy.array(jacobian)        
        
        return self._jacobian
    
    
    def _calc_adjusted_data(self, estimate):
        """
        Calculate the adjusted data vector based on the current estimate
        """
        
        if self._jacobian == None:
            
            self._jacobian = self._build_jacobian(estimate)
        
        adjusted = numpy.dot(self._jacobian, estimate)
        
        return adjusted
    
    
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
    
        
    def _build_first_deriv(self):
        """
        Compute the first derivative matrix of the model parameters.
        """
                
        # The number of derivatives there will be
        deriv_num = (self._nx - 1)*self._ny*self._nz + \
                    (self._ny - 1)*self._nx*self._nz + \
                    (self._nz - 1)*self._nx*self._ny
                
        first_deriv = numpy.zeros((deriv_num, self._nparams))
        
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
        
        return first_deriv
    
        
    def depth_weights(self, z0, power, normalize=True):
        """
        Calculate and return the depth weight matrix as in Li and Oldenburg 
        (1996)
        
        Parameters:
                
            z0: reference height
            
            power: decrease rate of the kernel
        """
        
        self._log.info("Building depth weights:")
        self._log.info("  z0 = %g" % (z0))
        self._log.info("  power = %g" % (power))
        self._log.info("  normalized = %s" % (str(normalize)))
                      
        weight = numpy.identity(self._nparams)
        
        dz = (self._mod_z2 - self._mod_z1)/self._nz
        
        depths = numpy.arange(self._mod_z1, self._mod_z2, dz, 'float')
        
        l = 0
        
        for depth in depths:
            
            for j in xrange(self._ny*self._nx):

                weight[l][l] = math.sqrt( \
                                1./(math.sqrt(depth + 0.5*dz + z0)**power))
                
                l += 1
        
        if normalize:
            
            weight = weight/weight.max()
        
        return weight
            
    
    def set_equality(self, x, y, z, value):
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
              

    def plot_adjustment(self, shape, title="Adjustment"):
        """
        Plot the original data plus the adjusted data with contour lines.
        """
        
        adjusted = self._calc_adjusted_data(self.mean)
        
        data_types = [self._gz, self._gxx, self._gxy, self._gxz, self._gyy, \
                      self._gyz, self._gzz]
        
        titles = ['$g_{z}$', '$g_{xx}$', '$g_{xy}$', '$g_{xz}$', '$g_{yy}$', \
                  '$g_{yz}$', '$g_{zz}$']
        
        i = 0
        
        for data in data_types:
            
            if data:                
            
                adj_matrix = numpy.reshape(adjusted[:len(data)], shape)
                
                # Remove this data set from the adjuted data
                adjusted = adjusted[len(data):]
                
                X = data.get_xgrid(*shape)
                
                Y = data.get_ygrid(*shape)
                
                vmin = min([adj_matrix.min(), data.array.min()])
                vmax = max([adj_matrix.max(), data.array.max()])
                
                pylab.figure()
                pylab.axis('scaled')
                pylab.title(title + titles[i])
                
                CS = pylab.contour(X, Y, adj_matrix, colors='r', \
                                   label="Adjusted", vmin=vmin, vmax=vmax)
                pylab.clabel(CS)
                
                CS = pylab.contour(X, Y, data.togrid(*X.shape), colors='b', \
                              label="Observed", vmin=vmin, vmax=vmax)
                
                pylab.clabel(CS)
                
                pylab.xlabel("Y")
                pylab.ylabel("X")
                
                pylab.xlim(Y.min(), Y.max())
                pylab.ylim(X.min(), X.max())
        
            i += 1
                        
        
    def plot_mean_layers(self):
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
                    
    
    def plot_std_layers(self):
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
                        
    
    def plot_mean(self):
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
        fig.scene.background = (0.1, 0.1, 0.1)
        fig.scene.camera.pitch(180)
        fig.scene.camera.roll(180)
        
        source = mlab.pipeline.add_dataset(grid)
        ext_grid = mlab.pipeline.extract_grid(source)        
        threshold = mlab.pipeline.threshold(ext_grid)
        axes = mlab.axes(threshold, nb_labels=self._nx+1, \
                         extent=[prism_xs[0], prism_xs[-1], \
                                 prism_ys[0], prism_ys[-1], \
                                 prism_zs[0], prism_zs[-1]])
        surf = mlab.pipeline.surface(axes, vmax=model.max(), vmin=model.min())
#        surf.actor.property.edge_visibility = 1
#        surf.actor.property.line_width = 1.5
        mlab.colorbar(surf, title="Density [g/cm^3]", orientation='vertical', \
                      nb_labels=10)
        
        
    def plot_stddev(self, title="Standard Deviation"):
        """
        Plot the standard deviation of the results in 3D using Mayavi
        """
        
        dx = (self._mod_x2 - self._mod_x1)/self._nx
        dy = (self._mod_y2 - self._mod_y1)/self._ny      
        dz = (self._mod_z2 - self._mod_z1)/self._nz
        
        prism_xs = numpy.arange(self._mod_x1, self._mod_x2 + dx, dx, 'float')
        prism_ys = numpy.arange(self._mod_y1, self._mod_y2 + dy, dy, 'float')
        prism_zs = numpy.arange(self._mod_z1, self._mod_z2 + dz, dz, 'float')
        
        std = numpy.reshape(self.stddev, (self._nz, self._ny, self._nx))*0.001
                        
        grid = tvtk.RectilinearGrid()
        grid.cell_data.scalars = std.ravel()
        grid.cell_data.scalars.name = 'Standard Deviation'
        grid.dimensions = (self._nx + 1, self._ny + 1, self._nz + 1)
        grid.x_coordinates = prism_xs
        grid.y_coordinates = prism_ys
        grid.z_coordinates = prism_zs
        
        fig = mlab.figure()
        fig.scene.background = (0.1, 0.1, 0.1)
        fig.scene.camera.pitch(180)
        fig.scene.camera.roll(180)       
        source = mlab.pipeline.add_dataset(grid)
        filter = mlab.pipeline.threshold(source)
        axes = mlab.axes(filter, nb_labels=self._nx+1, \
                         extent=[prism_xs[0], prism_xs[-1], \
                                 prism_ys[0], prism_ys[-1], \
                                 prism_zs[0], prism_zs[-1]])
        surf = mlab.pipeline.surface(axes, vmax=std.max(), vmin=std.min())
#        surf.actor.property.edge_visibility = 1
#        surf.actor.property.line_width = 1.5
        mlab.colorbar(surf, title="Standard Deviation [g/cm^3]", \
                      orientation='vertical', \
                      nb_labels=10)
        
        
    def dump(self, fname):
        """
        Dumps the mean result and standard deviation into file fname. 
        """
        
        file_obj = file(fname, 'w')
        
        file_obj.write("%g %g %d\n" % (self._mod_x1, self._mod_x2, self._nx))
        file_obj.write("%g %g %d\n" % (self._mod_y1, self._mod_y2, self._ny))
        file_obj.write("%g %g %d\n" % (self._mod_z1, self._mod_z2, self._nz))
        
        pylab.savetxt(file_obj, self.estimates)
        
        
    def load(self, fname):
        """
        Load saved estimates from file fname.
        """
        
        file_obj = file(fname, 'r')
        
        line = file_obj.readline()        
        self._mod_x1, self._mod_x2, self._nx = [float(v) \
                                                for v in line.split(' ')]
        
        line = file_obj.readline()        
        self._mod_y1, self._mod_y2, self._ny = [float(v) \
                                                for v in line.split(' ')]
        
        line = file_obj.readline()        
        self._mod_z1, self._mod_z2, self._nz = [float(v) \
                                                for v in line.split(' ')]
                
        self.estimates = pylab.loadtxt(file_obj)
        