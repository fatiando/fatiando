"""
Example script for doing the inversion of synthetic FTG data
"""

import pickle
import logging
log = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter())
log.addHandler(handler)
log.setLevel(logging.DEBUG)

import pylab
import numpy
from enthought.mayavi import mlab

from fatiando.inversion import pgrav3d
from fatiando.gravity import io
import fatiando.geometry
import fatiando.utils
from fatiando.visualization import residuals_histogram, plot_prism_mesh

# Load the synthetic data
gzz = io.load('gzz_data.txt')

data = {'gzz':gzz}

# Generate a model space mesh
mesh = fatiando.geometry.prism_mesh(x1=-800, x2=800, y1=-800, y2=800,
                                    z1=0, z2=1600, nx=8, ny=8, nz=8)

# Run the inversion
estimate, goals = pgrav3d.solve(data, mesh, initial=None, 
                                damping=10**(-10), 
                                smoothness=10**(-5),
                                compactness=3*10**(-3), epsilon=10**(-5),
                                lm_start=0.01)

pgrav3d.fill_mesh(estimate, mesh)

residuals = pgrav3d.residuals(data, estimate)

pylab.figure()
residuals_histogram(residuals)

# Load the synthetic model for comparison
synth_file = open('model.pickle')
synthetic = pickle.load(synth_file)
synth_file.close()

fig = mlab.figure()
fig.scene.background = (0.1, 0.1, 0.1)
fig.scene.camera.pitch(180)
fig.scene.camera.roll(180)

plot = plot_prism_mesh(synthetic, style='wireframe', label='Synthetic')

plot_prism_mesh(mesh, style='surface', label='Density')
axes = mlab.axes(plot, nb_labels=9, extent=[-800,800,-800,800,0,1600])

mlab.show()
