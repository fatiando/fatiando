"""
GravMag: Use an equivalent layer to reduce a magnetic total field anomaly to the
pole
"""
from fatiando import gravmag, gridder, utils, mesher
from fatiando.vis import mpl

# Make synthetic data
inc, dec = -10, 23
magdir = utils.ang2vec(1, inc, dec)
props = {'magnetization':10*magdir}
model = [mesher.Prism(400, 600, 300, 700, 200, 600, props)]
shape = (25, 25)
x, y, z = gridder.regular([0, 1000, 0, 1000], shape, z=0)
tf = utils.contaminate(gravmag.prism.tf(x, y, z, model, inc, dec), 10)
# Setup the layer
grid = mesher.PointGrid([-500, 1500, -500, 1500], 200, (50, 50))
# Estimate the magnetization intensity
data = [gravmag.eqlayer.TotalField(x, y, z, tf, inc, dec)]
intensity, predicted = gravmag.eqlayer.classic(data, grid, damping=0.0000000001)
residuals = tf - predicted[0]
# Convert the intensity to magnetization
magnetization = [i*magdir for i in intensity]
grid.addprop('magnetization', magnetization)
# Plot the layer and the fit
mpl.figure()
mpl.subplot(2, 1, 1)
mpl.axis('scaled')
mpl.title('Layer')
mpl.pcolor(grid.y, grid.x, intensity, grid.shape)
mpl.subplot(2, 1, 2)
mpl.axis('scaled')
mpl.title('Fit')
levels = mpl.contour(y, x, tf, shape, 12, color='r')
mpl.contour(y, x, predicted[0], shape, levels, color='k')
mpl.show()
