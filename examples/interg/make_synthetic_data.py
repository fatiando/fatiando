"""
Generate synthetic gravity profile from the relief of a basin
"""

import pickle

import numpy
import pylab

from fatiando.grav import io, synthetic
import fatiando.mesh
import fatiando.utils
import fatiando.stats
import fatiando.vis

# Get a logger for the script
log = fatiando.utils.get_logger()

# Make a synthetic model mesh 
mesh = fatiando.mesh.line_mesh(0, 5000, 100)
dx = mesh[0]['x2'] - mesh[0]['x1']
y1 = -1000.*dx
y2 = 1000.*dx
density = -500.
amp = 2000.
log.info("Generating smooth gaussian model")

for i, cell in enumerate(mesh.ravel()):
    
    x = 0.5*(cell['x1'] + cell['x2'])
        
    cell['value'] = density
    cell['y1'] = y1
    cell['y2'] = y2
    cell['z1'] = 0
    cell['z2'] = amp*(fatiando.stats.gaussian(x, 1500., 1000.) +
                      0.4*fatiando.stats.gaussian(x, 3500., 1000.))
        
    
modelfile = open("model.pickle", 'w')
pickle.dump(mesh, modelfile)
modelfile.close()

# Extract the topography from the model mesh and set a height above it for the
# measurements
topo = fatiando.mesh.extract_key('z1', mesh)
height = -1*topo + 1

topofile = open('topo.pickle', 'w')
pickle.dump(topo, topofile)
topofile.close()

# Calculate the gravitational effect of the model
gz = synthetic.from_prisms(mesh, x1=0, x2=5000, y1=0, y2=0, nx=100, ny=1, 
                           height=height, field='gz')

# Contaminate it with gaussian noise and save
error = 0.1
gz['value'] = fatiando.utils.contaminate(gz['value'], stddev=error, 
                                                percent=False)
gz['error'] = error*numpy.ones_like(gz['value'])
io.dump("gzprofile.txt", gz)

# Plot the model and gravity effect
pylab.figure(figsize=(10,8))
pylab.suptitle("Synthetic Gravity Data", fontsize=14)
pylab.subplots_adjust(hspace=0.1)

pylab.subplot(2,1,1)
pylab.plot(gz['x'], gz['value'], '.-k', label=r"Synthetic $g_z$")
pylab.ylabel("mGal")
pylab.legend(loc='lower right', shadow=True)

pylab.subplot(2,1,2)
fatiando.vis.plot_2d_interface(mesh, 'z2', style='-k', linewidth=1, fill=mesh, 
                               fillkey='z1', fillcolor='gray', alpha=0.5,
                               label='Basin Model')
pylab.ylim(1.2*amp, -300)
pylab.xlabel("X [m]")
pylab.ylabel("Depth [m]")
pylab.text(2500, 500, r"$\Delta\rho = -500\ kg.m^{-3}$", fontsize=16,
           horizontalalignment='center')
pylab.legend(loc='lower right', shadow=True)

pylab.show()
