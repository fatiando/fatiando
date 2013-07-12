"""
GravMag: Use an equivalent layer to reduce a magnetic total field anomaly to the
pole
"""
from fatiando import gravmag, gridder, utils, mesher
from fatiando.vis import mpl

# Make synthetic data
inc, dec = -60, 23
props = {'magnetization':10}
model = [mesher.Prism(-500, 500, -1000, 1000, 500, 4000, props)]
shape = (25, 25)
x, y, z = gridder.regular([-5000, 5000, -5000, 5000], shape, z=0)
tf = utils.contaminate(gravmag.prism.tf(x, y, z, model, inc, dec), 5)
# Setup the layer
grid = mesher.PointGrid([-7000, 7000, -7000, 7000], 1000, (50, 50))
# Estimate the magnetization intensity
data = [gravmag.eqlayer.TotalField(x, y, z, tf, inc, dec)]
# Need to apply enough damping so that won't try to fit the error as well
intensity, predicted = gravmag.eqlayer.classic(data, grid, damping=0.02)
grid.addprop('magnetization', intensity)
residuals = tf - predicted[0]
print "Residuals:"
print "mean:", residuals.mean()
print "stddev:", residuals.std()
# Plot the layer and the fit
mpl.figure(figsize=(14,4))
mpl.subplot(1, 3, 1)
mpl.axis('scaled')
mpl.title('Layer (A/m)')
mpl.pcolor(grid.y, grid.x, grid.props['magnetization'], grid.shape)
mpl.subplot(1, 3, 2)
mpl.axis('scaled')
mpl.title('Fit (nT)')
levels = mpl.contour(y, x, tf, shape, 15, color='r')
mpl.contour(y, x, predicted[0], shape, levels, color='k')
mpl.subplot(1, 3, 3)
mpl.title('Residuals (nT)')
mpl.hist(residuals, bins=10)
mpl.show()
# Now I can forward model the layer at the south pole and check against the
# true solution of the prism
tfpole = gravmag.prism.tf(x, y, z, model, -90, 0)
tfreduced = gravmag.sphere.tf(x, y, z, grid, -90, 0)
mpl.figure(figsize=(14,4))
mpl.subplot(1, 2, 1)
mpl.axis('scaled')
mpl.title('True (red) | Reduced (black)')
levels = mpl.contour(y, x, tfpole, shape, 12, color='r')
mpl.contour(y, x, tfreduced, shape, levels, color='k')
mpl.subplot(1, 2, 2)
mpl.title('True - reduced (nT)')
mpl.hist(tfpole - tfreduced, bins=10)
mpl.show()
