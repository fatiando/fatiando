"Run an example tomography"

import sys
sys.path.append('/home/leo/src/fatiando/dev')

import pylab
import numpy

from fatiando import vis, utils
from fatiando import mesh as meshgen
from fatiando.seismo import synthetic, tomosim

# Make a logger for the script
log = utils.get_logger()
log.info(utils.header())

# Load the image model and convert it to a velocity model
model = synthetic.vel_from_image('mickey.jpg', vmax=5., vmin=1.)

# Shoot rays through the model simulating an X-ray tomography configuration
data = synthetic.shoot_cartesian_straight(model, src_n=70, rec_n=30, type='xray')

# Make the model space mesh
ny, nx = model.shape
mesh = meshgen.square_mesh(x1=0, x2=nx, y1=0, y2=ny, nx=nx, ny=ny)

# Solve
init = 1.5*numpy.ones(mesh.size)
results = tomosim.solve(data, mesh, init, damp=1*10**(-1), lmstart=0.1, tol=10**(-4))

# Put the result in the mesh (for plotting)
meshgen.fill(results['estimate'], mesh)

# Plot the synthetic model and inversion results
pylab.figure(figsize=(12,8))
pylab.suptitle("X-ray simulation", fontsize=14)
vmin = min(results['estimate'].min(), model.min())
vmax = max(results['estimate'].max(), model.max())
vmin = model.min()
vmax = model.max()

pylab.subplot(2,2,1)
pylab.axis('scaled')
pylab.title("Synthetic velocity model")
ax = pylab.pcolor(model, cmap=pylab.cm.jet, vmin=vmin, vmax=vmax)
cb = pylab.colorbar()
cb.set_label("Velocity")
pylab.xlim(0, nx)
pylab.ylim(0, ny)

pylab.subplot(2,2,2)
pylab.axis('scaled')
pylab.title("Inversion result")
vis.plot_square_mesh(mesh, vmin=vmin, vmax=vmax)
cb = pylab.colorbar()
cb.set_label("Velocity")
pylab.xlim(0, nx)
pylab.ylim(0, ny)

#~ pylab.subplot(2,2,3)
#~ pylab.title("Goal function per iteration")
#~ pylab.plot(results['goal_p_it'], '-k')

pylab.subplot(2,2,4)
pylab.title("Histogram of residuals")
vis.residuals_histogram(results['residuals'], nbins=len(results['residuals'])/10)
pylab.xlabel("Travel time residuals")
pylab.ylabel("Number of occurences")

pylab.show()
