# Copyright 2010 The Fatiando a Terra Development Team
#
# This file is part of Fatiando a Terra.
#
# Fatiando a Terra is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Fatiando a Terra is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Fatiando a Terra.  If not, see <http://www.gnu.org/licenses/>.
"""
Gravity inversion of an interface using right rectangular prisms.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 05-Jul-2010'

import time
import logging

import pylab
import numpy

import fatiando
from fatiando.inversion.gradientsolver import GradientSolver
from fatiando.directmodels.gravity import prism as prism_gravity


logger = logging.getLogger('InterG2D')
logger.setLevel(logging.DEBUG)
logger.addHandler(fatiando.default_log_handler)



class InterG2D(GradientSolver):
    """
    2D gravity inversion of an interface using right rectangular prisms
    
    Parameters:
        
        x1, x2: boundaries of the model space
        
        nx: number of prisms into which the model space will be cut
                    
        gz: instance of fatiando.data.gravity.VerticalGravity holding the
            vertical gravity data
            
        gxx, gxy, gxz, gyy, gyz, gzz: instances of 
            fatiando.data.gravity.TensorComponent holding each a respective
            gravity gradient tensor component data
            
    Note: at least of one gz, gxx, gxy, gxz, gyy, gyz, or gzz must be 
    provided
    """
    
    def __init__(self, x1, x2, nx, dens, gz=None, gxx=None, gxy=None, \
                 gxz=None, gyy=None, gyz=None, gzz=None):
        
        GradientSolver.__init__(self)
        
        assert gz != None or gxx != None or gxy != None or gxz != None or \
            gyy != None or gyz != None or gzz != None, \
            "Provide at least one of gz, gxx, gxy, gxz," + \
            " gyy, gyz, or gzz. Can't do the inversion without data!"
        
        # Data parameters
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
        self._nx = nx
        self._nparams = nx
        self._dens = dens
        
        # The logger for this class
        self._log = logging.getLogger('InterG2D')
        
        self._log.info("Model space discretization: %d parameters" % (nx))
    
    
    def _build_jacobian(self, estimate):
        """
        Make the Jacobian matrix of the function of the parameters.
        """
        
        assert estimate != None, "Can't use solve_linear method. " + \
            "This is a non-linear problem."
        
        delta = 0.1
        
        dx = (self._mod_x2 - self._mod_x1)/self._nx
        
        prism_xs = numpy.arange(self._mod_x1, self._mod_x2, dx, 'float')
        
        jacobian = []
        
        data_types = [self._gz, self._gxx, self._gxy, self._gxz, self._gyy, \
                      self._gyz, self._gzz]
        
        calc_types = [prism_gravity.gz, prism_gravity.gxx, prism_gravity.gxy, \
                      prism_gravity.gxz, prism_gravity.gyy, prism_gravity.gyz, \
                      prism_gravity.gzz]
        
        for t in xrange(len(data_types)):
            
            if data_types[t]:
                
                for i in xrange(len(data_types[t])):
                    
                    xp, yp, zp = data_types[t].loc(i)
                    
                    line = numpy.zeros(self._nx)
                    
                    l = 0                                        
                        
                    for x in prism_xs:
                            
                        tmp_plus = calc_types[t](self._dens, \
                                    x, x + dx, \
                                    -10**(6), 10**(6), \
                                    0, estimate[l] + delta, xp, yp, zp) 
                        
                        tmp_minus = calc_types[t](self._dens, \
                                    x, x + dx, \
                                    -10**(6), 10**(6), \
                                    0, estimate[l] - delta, xp, yp, zp)
                        
                        line[l] = (tmp_plus - tmp_minus)/(2*delta)
                        
                        l += 1
                            
                    jacobian.append(line)
                    
        jacobian = numpy.array(jacobian)
                
        return jacobian
            
    
    def _calc_adjusted_data(self, estimate):
        """
        Calculate the adjusted data vector based on the current estimate
        """
        
        dx = (self._mod_x2 - self._mod_x1)/self._nx
        
        prism_xs = numpy.arange(self._mod_x1, self._mod_x2, dx, 'float')
        
        adjusted = numpy.array([])
        
        data_types = [self._gz, self._gxx, self._gxy, self._gxz, self._gyy, \
                      self._gyz, self._gzz]
        
        calc_types = [prism_gravity.gz, prism_gravity.gxx, prism_gravity.gxy, \
                      prism_gravity.gxz, prism_gravity.gyy, prism_gravity.gyz, \
                      prism_gravity.gzz]
        
        for t in xrange(len(data_types)):
            
            if data_types[t]:
                
                tmp = numpy.zeros(len(data_types[t]))
                
                for i in xrange(len(data_types[t])):
                    
                    xp, yp, zp = data_types[t].loc(i)
                    
                    l = 0
                        
                    for x in prism_xs:
                        
                        tmp[i] += calc_types[t](self._dens, \
                                    x, x + dx, \
                                    -10**(6), 10**(6), \
                                    0, estimate[l], xp, yp, zp)
                        
                        l += 1
                            
                adjusted = numpy.append(adjusted, tmp)
                
        return adjusted
        
        
    def _build_first_deriv(self):
        """
        Compute the first derivative matrix of the model parameters.
        """
                
        # The number of derivatives there will be
        deriv_num = (self._nx - 1)
        
        first_deriv = numpy.zeros((deriv_num, self._nparams))
                
        # Derivatives in the x direction   
        for i in range(self._nx - 1):                
            
            first_deriv[i][i] = -1
            
            first_deriv[i][i + 1] = 1
                
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
    
    
    def split(self, factor):
        """
        Split the mean solution into factor*nx prisms.
        
        Parameters:
        
            factor: into how many prisms each prism will be split into
            
        Returns the new solution with factor*nx elements.
        """
                
        new = []
        
        mean = self.mean
        
        for i in xrange(self._nx):
            
            for j in xrange(factor):
                
                new.append(mean[i])
            
        return numpy.array(new)            
   
        
    def set_equality(self, x, z):
        """
        Set an equality constraint to hold the prism with x coordinate at depth 
        z
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
        
        prism_xs = numpy.arange(self._mod_x1, self._mod_x2, dx, 'float')
                                    
        for l in xrange(len(prism_xs)):
            
            if x >= prism_xs[l] and x <= prism_xs[l] + dx:
                
                p_ref.append(z)
                
                tmp = numpy.zeros(self._nx)
                
                tmp[l] = 1
                
                D.append(tmp)
                    
        self._equality_matrix = numpy.array(D)
        
        self._equality_values = numpy.array(p_ref)
        
        
    def plot_mean(self, true_x=None, true_z=None, \
                  title="Mean inversion result"):
        """
        Plot the mean solution
        """
                        
        dx = (self._mod_x2 - self._mod_x1)/self._nx
        
        prism_xs = numpy.arange(self._mod_x1, self._mod_x2 + dx, dx, 'float')
                    
        mean = numpy.array(self.mean)
        
        model = []
        
        x = []
                
        for i in xrange(len(mean)):
            
            model.append(mean[i])
            model.append(mean[i])
            
            x.append(prism_xs[i])
            x.append(prism_xs[i] + dx)
                
        pylab.figure()
        pylab.title(title)
        pylab.plot(x, model, '-k', label="Mean Solution", linewidth=2)
        
        vmax = max(model)
        vmin = min(model)
        
        if true_x != None and true_z != None:
            
            pylab.plot(true_x, true_z, '-r', label="True Model")
            
            vmax = numpy.append(true_z, vmax).max()
            vmin = numpy.append(true_z, vmin).min()
        
        for estimate in self.estimates:
                  
            model = []
                                
            for i in xrange(len(estimate)):
                
                model.append(estimate[i])
                model.append(estimate[i])
                    
            pylab.plot(x, model, '-b')
            
            vmax = numpy.append(model, vmax).max()
            vmin = numpy.append(model, vmin).min()
                
        pylab.xlabel("Position X [m]")
        pylab.ylabel("Depth [m]")
                    
        pylab.legend(prop={'size':9})
        
        pylab.xlim(prism_xs.min(), prism_xs.max())
        pylab.ylim(1.2*vmax, 0)
            
                        
    def plot_std(self, title="Result Standard Deviation", cmap=pylab.cm.jet):
        """
        Plot the standard deviation of the model parameters.
        """
        
        dx = (self._mod_x2 - self._mod_x1)/self._nx
        
        prism_xs = numpy.arange(self._mod_x1, self._mod_x2 + dx, dx, 'float')
                
        stds = numpy.array(self.stddev)
        
        model = []
        
        x = []
                
        for i in xrange(len(stds)):
            
            model.append(stds[i])
            model.append(stds[i])
            
            x.append(prism_xs[i])
            x.append(prism_xs[i] + dx)
        
        pylab.figure()
        pylab.title(title)
        pylab.plot(x, model, '-k')    
                
        pylab.xlabel("Position X [m]")
        pylab.ylabel("Standard Deviation [m]")
        
        pylab.xlim(prism_xs.min(), prism_xs.max())
        pylab.ylim(1.2*stds.min(), 1.2*stds.max())


    def plot_adjustment(self, title="Adjustment"):
        """
        Plot the original data plus the adjusted data
        """
        
        adjusted = self._calc_adjusted_data(self.mean)
        
        data_types = [self._gz, self._gxx, self._gxy, self._gxz, self._gyy, \
                      self._gyz, self._gzz]
        
        titles = [' $g_{z}$', ' $g_{xx}$', ' $g_{xy}$', ' $g_{xz}$', \
                  ' $g_{yy}$', ' $g_{yz}$', ' $g_{zz}$']
        
        i = 0
        
        for data in data_types:
            
            if data:                
                                                            
                x = data.get_xarray()
                                
                pylab.figure()
                pylab.title(title + titles[i])
                
                pylab.plot(x, adjusted[:len(data)], '.-r', label="Adjusted")
                
                pylab.plot(x, data.array, '.-k', label="Observed")
                    
                pylab.legend(prop={'size':9})
                
                pylab.xlabel("Position [m]")
                pylab.ylabel(titles[i] + " Eotvos")
                
                pylab.xlim(x.min(), x.max())
                
                # Remove this data set from the adjuted data
                adjusted = adjusted[len(data):]
        
            i += 1
