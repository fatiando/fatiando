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
import fatiando.stats
import fatiando.vis


# Make a synthetic model mesh 
mesh = fatiando.geometry.line_mesh(0, 5000, 100)
dx = mesh[0]['x2'] - mesh[0]['x1']
y1 = -10.*dx
y2 = 10.*dx
density = -500.
amp = 2000.
log.info("Generating smooth gaussian model")

for cell in mesh.ravel():
    
    x = 0.5*(cell['x1'] + cell['x2'])
        
    cell['value'] = density
    cell['y1'] = y1
    cell['y2'] = y2
    cell['z1'] = 0
    cell['z2'] = amp*fatiando.stats.gaussian(x, 2500., 1000.)    
    
modelfile = open("model.pickle", 'w')
pickle.dump(mesh, modelfile)
modelfile.close()

# Calculate the effect of the model
gz = synthetic.from_prisms(mesh, 0, 5000, 0, 0, 200, 1, height=1, field='gz')

# Contaminate it with gaussian noise and save
error = 0.1
gz['value'] = fatiando.utils.contaminate(gz['value'], stddev=error, 
                                                percent=False)
gz['error'] = error*numpy.ones_like(gz['value'])
io.dump("gzprofile.txt", gz)

# Plot the model and gravity effect
pylab.figure()

pylab.subplot(2,1,1)
pylab.title("Synthetic gz")
pylab.plot(gz['x'], gz['value'], '.-k')
pylab.ylabel("mGal")

pylab.subplot(2,1,2)
pylab.title("Interface")
fatiando.vis.plot_2d_interface(mesh, 'z2', style='-k', linewidth=1, 
                               label='Interface')
pylab.ylim(1.2*amp, -200)
pylab.xlabel("X [m]")
pylab.ylabel("Depth [m]")

pylab.show()
