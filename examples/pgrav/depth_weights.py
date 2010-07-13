import logging

import numpy
import pylab

from fatiando.directmodels.gravity import prism
from fatiando.geoinv.lmsolver import LMSolver
import fatiando

logger = logging.getLogger('depthweights')       
logger.setLevel(logging.DEBUG)
logger.addHandler(fatiando.default_log_handler)

class DepthWeights(LMSolver):
    """
    Solves for the coeficients of the depth weighing function
    """


    def __init__(self, zs, dz, gzz):
        """
        Parameters:
        
            zs: array with the top depths of the model layers
            
            dz: model cell size in the z direction 
        """
        
        LMSolver.__init__(self)
        
        assert len(zs) == len(gzz), "Must have same number of depths and gzzs"
        
        self._zs = zs
        self._dz = dz
        self._gzz = gzz
                
        self._nparams = 3
        
        self._log = logging.getLogger('depthweights')
        
        
    def _build_jacobian(self, estimate):
        """
        Make the Jacobian matrix of the function of the parameters.
        'estimate' is the the point in the parameter space where the Jacobian
        will be evaluated.
        """
        
        jacobian = []
        
        for z in self._zs:
            
            tmp1 = 1./((z + 0.5*self._dz + estimate[1])**estimate[2])
            
            tmp2 = -estimate[0]*estimate[2]/( \
                            (z + 0.5*self._dz + estimate[1])**(estimate[2] + 1))
            
            tmp3 = -estimate[0]/((z + 0.5*self._dz + estimate[1])**estimate[2])
            
            jacobian.append([tmp1, tmp2, tmp3])
            
        return numpy.array(jacobian)        
            
    
    def _calc_adjusted_data(self, estimate):
        """
        Calculate the adjusted data vector based on the current estimate
        """
        
        adjusted = numpy.zeros(len(self._zs))
        
        for i in xrange(len(self._zs)): 
        
            adjusted[i] = estimate[0]/(\
                        (self._zs[i] + 0.5*self._dz + estimate[1])**estimate[2])
            
        return adjusted
            
        
    def _build_first_deriv(self):
        """
        Compute the first derivative matrix of the model parameters.
        """
        
        # Raise an exception if the method was raised without being implemented
        raise NotImplementedError, \
            "_build_first_deriv was called before being implemented"
            
            
    def _get_data_array(self):
        """
        Return the data in a Numpy array so that the algorithm can access it
        in a general way
        """        
        
        return numpy.array(self._gzz)
                           
            
    def _get_data_cov(self):
        """
        Return the data covariance in a 2D Numpy array so that the algorithm can
        access it in a general way
        """        
        
        return numpy.identity(len(self._zs))
        
                                      


def main():
    
    logging.basicConfig()
    
    from sys import stderr
            
    zmax = 1200
    zmin = 200
    
    dz = 200
    dx = 200
    dy = 200
    
    zs = numpy.arange(zmin, zmax, dz, dtype='float')
    
    gzz = []
    
    beta = 10**(-15)
    
    for i in xrange(len(zs)):
        
        tmp = prism.gzz(1., -0.5*dx, 0.5*dx, -0.5*dy, 0.5*dy, \
                        zs[i], zs[i] + dz, 0., 0., -150.)
        
#        if i == 0:
#            
#            gzz_surf = tmp
#            
#        tmp = 1./(gzz_surf - tmp + beta)
        
        gzz.append(tmp)
        
    gzz = numpy.array(gzz)    
    
    solver = DepthWeights(zs=zs, dz=dz, gzz=gzz)
        
    solver.solve(damping=0, smoothness=0, curvature=0, sharpness=0, \
                 equality=0, initial_estimate=[1,150,3], apriori_var=0, \
                 contam_times=0, max_it=100, max_lm_it=20, lm_start=10, lm_step=10)
        
    ws = solver._calc_adjusted_data(solver.mean)
    
    stderr.write("\nw0=%g, \\" % (solver.mean[0]))
    stderr.write("\nz0=%g, \\" % (solver.mean[1]))
    stderr.write("\npower=%g, \\" % (solver.mean[2]))
        
    pylab.figure()
    pylab.title("Adjustment")
    pylab.plot(gzz, zs + 0.5*dz, '-b', label='$g_{zz}$')
    pylab.plot(ws, zs + 0.5*dz, '-r', label='Weights')
    pylab.legend(loc='lower right')
    pylab.xlabel('$g_{zz}$')
    pylab.ylabel('Depth')
    pylab.ylim(zmax,0)
    pylab.show()
    
    
if __name__ == '__main__':
    
    main()