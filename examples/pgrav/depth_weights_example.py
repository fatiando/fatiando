import logging
logging.basicConfig()

import pylab

from fatiando.geoinv.pgrav import PGrav3D, DepthWeightsCalculator


class MockData():
    """
    Fool PGrav3D into thinking we gave it real data
    """
    
    def __init__(self):
        
        pass
    
    def __len__(self):
        
        return 0
    
    
solver = PGrav3D(x1=0, x2=1000, y1=0, y2=1000, z1=0, z2=1000, \
                 nx=10, ny=10, nz=10, \
                 gzz=MockData())
    
dwsolver = DepthWeightsCalculator(pgrav_solver=solver, height=150)

dwsolver.solve_lm(initial=[150, 3], data_variance=0.0001**2, contam_times=5, \
                  equality=0, \
                  lm_start=1, lm_step=10, it_p_step=20, max_it=100)

z0, power = dwsolver.mean

print "z0:", z0, " +-", dwsolver.stddev[0]
print "power:", power, " +-", dwsolver.stddev[1]

dwsolver.plot_adjustment()

pylab.show()