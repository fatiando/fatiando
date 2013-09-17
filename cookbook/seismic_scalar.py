"""
Seismic: 2D finite difference simulation of scalar wave propagation
"""

from fatiando import seismic
import numpy as np

mvel = np.zeros((100,100))+3000

# Set the parameters of the finite difference grid
shape = np.shape(mvel)
spacing = 4. # marmousi spacing
#area = (0, spacing[1]*shape[1], 0, spacing[0]*shape[0])

vmax = mvel.max()
vmin = mvel.min()

# source frequency ? waveleght w = 0.0 second
# 1/w = 2 Hz
#fc=1./0.06
fc=vmin/(10.*spacing) # from dave hale code (Madagascar)

# Make a wave source from a mexican hat wavelet, delay=3.5*wavelength
sources = [seismic.wavefd.MexHatSource(2, 1250, 100, 1./fc, delay=3.5/fc)]


# Get the iterator. This part only generates an iterator object. The actual
# computations take place at each iteration in the for loop below
dt = 0.004
maxit = int(3.0/dt) # 3.0 seconds marmousi shots
timesteps = seismic.wavefd.scalar(spacing, shape, mvel, dt, maxit,
    sources, padding=0.2)

R = dt*vmax/spacing
print "fc :", fc, " maxit ", maxit
print "vmin : ", vmin, " vmax : ", vmax
print "R : ", R, "  of allowed < : ", np.sqrt(3./8.)  
print "Dt : ",  dt, "  of allowed < :", np.sqrt(3./8.)*spacing/vmax
# Dave Hale criteria for dt
print "Dave Hale dt", spacing/(2*vmax)
print "Points by wavelenght: ", vmin/fc/spacing, " recommended > 5 "
print "Is there spatial Alias (based on frequency)? ", (vmin/fc < 2*spacing)