"""
Generate synthetic gravity profile from a relief
"""

import pickle
import logging
log = logging.getLogger()
shandler = logging.StreamHandler()
shandler.setFormatter(logging.Formatter())
log.addHandler(shandler)
log.setLevel(logging.DEBUG)

import numpy
import pylab

from fatiando.gravity import io, synthetic
import fatiando.geometry
import fatiando.utils
import fatiando.vis

import math

# Define synthetic the model
mesh = fatiando.geometry.line_mesh(0, 5000, 100)
y1 = -10*
y2 = 10*dx
density = -500.


bottoms = []
prisms = []

# Make a smooth gaussian model
log.info("Generating smooth gaussian model:")

amplitude = 1000.
stddev = 1000.

log.info("  amplitude: %g" % (amplitude))
log.info("  dispersion: %g" % (stddev))

for x in xs:
    
    bottom = amplitude*math.exp(-1*((x + 0.5*dx - 0.5*(x1 + x2))**2)/ \
                                   (stddev**2))
    
    bottoms.append(bottom)
    
    prism = {'density':density, 'x1':x, 'x2':x + dx, 'y1':y1, 'y2':y2, 'z1':0, 
             'z2':bottom}
    
    prisms.append(prism)

prisms = numpy.array(prisms)



# Calculate the effect of the model
prof_nx = 100
prof_dx = float(x2 - x1)/(prof_nx - 1)
profile_xs = numpy.arange(x1, x2, prof_dx)
X = numpy.array([profile_xs])
Y = numpy.zeros_like(X)

data = TensorComponent('z')
data.synthetic_prism(prisms=prisms, X=X, Y=Y, z=-100, \
                     stddev=0.005, percent=True)
data.dump('gzprofile.txt')


# Plot the model and gravity effect
pylab.figure()

pylab.subplot(2, 1, 1)
pylab.title("$g_z$")

pylab.plot(profile_xs, data.array, '.-r')

pylab.ylabel("[mGal]")

pylab.xlim(xs[0], xs[-1])

pylab.subplot(2, 1, 2)

plot_xs = []
plot_botts = []

for x, bottom in zip(xs, bottoms):
    
    plot_xs.append(x)
    plot_xs.append(x + dx)
    
    plot_botts.append(bottom)
    plot_botts.append(bottom)

pylab.plot(plot_xs, plot_botts, '-k')

pylab.xlim(xs[0], xs[-1])
pylab.ylim(1.1*max(bottoms), 0)

pylab.xlabel("x [m]")
pylab.ylabel("depth [m]")

model = numpy.array([plot_xs, plot_botts]).T
pylab.savetxt('true_model.txt', model)

pylab.show()
