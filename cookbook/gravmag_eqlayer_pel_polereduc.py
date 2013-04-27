"""
GravMag: Use the Polynomial Equivalent Layer to reduce a magnetic total field
anomaly to the pole
"""
from fatiando import gravmag, gridder, utils, mesher
from fatiando.vis import mpl

# Make synthetic data
inc, dec = -60, 23
props = {'magnetization':10}
model = [mesher.Prism(-500, 500, -1000, 1000, 500, 4000, props)]
shape = (50, 50)
x, y, z = gridder.regular([-5000, 5000, -5000, 5000], shape, z=-150)
tf = utils.contaminate(gravmag.prism.tf(x, y, z, model, inc, dec), 5)
# Setup the layer
grid = mesher.PointGrid([-5000, 5000, -5000, 5000], 200, (100, 100))
# Wrape the data
data = [gravmag.eqlayer.TotalField(x, y, z, tf, inc, dec)]
# Calculate the magnetization intensity
# PEL returns the matrices it computes so that you can re-calculate with
# different smoothness and damping at very low cost
intensity, matrices = gravmag.eqlayer.pel(data, grid, (20, 20), degree=1,
    smoothness=10.**-2)
grid.addprop('magnetization', intensity)
# Compute the predicted data and the residuals
predicted = gravmag.sphere.tf(x, y, z, grid, inc, dec)
residuals = tf - predicted
print "Residuals:"
print "mean:", residuals.mean()
print "stddev:", residuals.std()
# Plot the layer and the fit
mpl.figure(figsize=(15, 4))
mpl.subplot(1, 3, 1)
mpl.axis('scaled')
mpl.title('Layer (A/m)')
mpl.pcolor(grid.y, grid.x, grid.props['magnetization'], grid.shape)
mpl.colorbar()
mpl.m2km()
mpl.subplot(1, 3, 2)
mpl.axis('scaled')
mpl.title('Fit (nT)')
levels = mpl.contour(y, x, tf, shape, 15, color='r')
mpl.contour(y, x, predicted, shape, levels, color='k')
mpl.m2km()
mpl.subplot(1, 3, 3)
mpl.title('Residuals (nT)')
mpl.hist(residuals, bins=10)
mpl.show()
# Now I can forward model the layer at the south pole and 500 m above the
# original data. Check against the true solution of the prism
tfpole = gravmag.prism.tf(x, y, z - 500, model, -90, 0)
tfreduced = gravmag.sphere.tf(x, y, z - 500, grid, -90, 0)
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
