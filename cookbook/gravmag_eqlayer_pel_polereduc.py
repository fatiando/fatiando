"""
GravMag: Use the Polynomial Equivalent Layer to reduce a magnetic total field
anomaly to the pole
"""
from fatiando.gravmag import prism, sphere
from fatiando.gravmag.eqlayer import PELTotalField, PELSmoothness
from fatiando import gridder, utils, mesher
from fatiando.vis import mpl

# Make synthetic data
inc, dec = -60, 23
props = {'magnetization': 10}
model = [mesher.Prism(-500, 500, -1000, 1000, 500, 4000, props)]
shape = (50, 50)
x, y, z = gridder.regular([-5000, 5000, -5000, 5000], shape, z=-150)
tf = utils.contaminate(prism.tf(x, y, z, model, inc, dec), 5, seed=0)
# Setup the layer
layer = mesher.PointGrid([-5000, 5000, -5000, 5000], 200, (100, 100))
# Estimate the density using the PEL (it is faster and more memory efficient
# than the traditional equivalent layer).
windows = (20, 20)
degree = 1
misfit = PELTotalField(x, y, z, tf, inc, dec, layer, windows, degree)
regul = PELSmoothness(layer, windows, degree)
# Apply a smoothness constraint to the borders of the equivalent layer windows
# to avoid gaps in the physical property distribution
solver = (misfit + 1e-15*regul).fit()
# Add the estimated density distribution to the layer object for plotting and
# forward modeling
layer.addprop('magnetization', solver.estimate_)
residuals = solver[0].residuals()
print("Residuals:")
print("mean:", residuals.mean())
print("stddev:", residuals.std())

# Now I can forward model the layer at the south pole and 500 m above the
# original data. Check against the true solution of the prism
tfpole = prism.tf(x, y, z - 500, model, -90, 0)
tfreduced = sphere.tf(x, y, z - 500, layer, -90, 0)

mpl.figure(figsize=(15, 4))
mpl.subplot(1, 3, 1)
mpl.axis('scaled')
mpl.title('Layer (A/m)')
mpl.pcolor(layer.y, layer.x, layer.props['magnetization'], layer.shape)
mpl.colorbar()
mpl.m2km()
mpl.subplot(1, 3, 2)
mpl.axis('scaled')
mpl.title('Fit (nT)')
levels = mpl.contour(y, x, tf, shape, 15, color='r')
mpl.contour(y, x, solver[0].predicted(), shape, levels, color='k')
mpl.m2km()
mpl.subplot(1, 3, 3)
mpl.title('Residuals (nT)')
mpl.hist(residuals, bins=10)

mpl.figure(figsize=(10, 4))
mpl.subplot(1, 2, 1)
mpl.axis('scaled')
mpl.title('True (red) | Reduced (black)')
levels = mpl.contour(y, x, tfpole, shape, 12, color='r')
mpl.contour(y, x, tfreduced, shape, levels, color='k')
mpl.m2km()
mpl.subplot(1, 2, 2)
mpl.title('True - reduced (nT)')
mpl.axis('scaled')
mpl.pcolor(y, x, tfpole - tfreduced, shape)
mpl.colorbar()
mpl.m2km()
mpl.show()
