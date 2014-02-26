"""
GravMag: Fit an equivalent layer to gravity and gravity gradient data
"""
from fatiando.gravmag import prism, sphere
from fatiando.gravmag.eqlayer import EQLGravity
from fatiando.inversion.regularization import Damping
from fatiando import gridder, utils, mesher
from fatiando.vis import mpl

# Make synthetic data
props = {'density':1000}
model = [mesher.Prism(-500, 500, -1000, 1000, 500, 4000, props)]
area = [-5000, 5000, -5000, 5000]
x1, y1, z1 = gridder.scatter(area, 100, z=0)
shape = (25, 25)
x2, y2, z2 = gridder.regular(area, shape, z=-150)
gz = utils.contaminate(prism.gz(x1, y1, z1, model), 0.1)
gzz = utils.contaminate(prism.gzz(x2, y2, z2, model), 1)
# Setup the layer
layer = mesher.PointGrid([-6000, 6000, -6000, 6000], 500, (50, 50))
# Estimate the density
# Need to apply enough damping so that won't try to fit the error as well
solver = (EQLGravity(x1, y1, z1, gz, layer)
          + EQLGravity(x2, y2, z2, gzz, layer, field='gzz')
          + 10**-24*Damping(layer.size))
solver.fit()
layer.addprop('density', solver.estimate_)

# Plot the layer and the fit
mpl.figure(figsize=(14, 4))
mpl.subplot(1, 3, 1)
mpl.axis('scaled')
mpl.title('Layer (kg.m^-3)')
mpl.pcolor(layer.y, layer.x, layer.props['density'], layer.shape)
mpl.colorbar()
mpl.m2km()
mpl.subplot(1, 3, 2)
mpl.axis('scaled')
mpl.title('Fit gz (mGal)')
levels = mpl.contour(y1, x1, gz, shape, 15, color='r', interp=True)
mpl.contour(y1, x1, solver.predicted()[0], shape, levels, color='k',
            interp=True)
mpl.m2km()
mpl.subplot(1, 3, 3)
mpl.axis('scaled')
mpl.title('Fit gzz (Eotvos)')
levels = mpl.contour(y2, x2, gzz, shape, 15, color='r')
mpl.contour(y2, x2, solver.predicted()[1], shape, levels, color='k')
mpl.m2km()
mpl.show()

# Now I can forward model gz using my layer to produce an integrated map in a
# much denser region
x, y, z = gridder.regular(area, shape, z=0)
gz_layer = sphere.gz(x, y, z, layer)
gz_true = prism.gz(x, y, z, model)

mpl.figure()
mpl.subplot(1, 2, 1)
mpl.axis('scaled')
mpl.title('True (red) | Layer (black)')
levels = mpl.contour(y, x, gz_true, shape, 12, color='r')
mpl.contour(y, x, gz_layer, shape, levels, color='k')
mpl.m2km()
mpl.subplot(1, 2, 2)
mpl.axis('scaled')
mpl.title('True (red) | Interpolation (black)')
levels = mpl.contour(y, x, gz_true, shape, 12, color='r')
mpl.contour(y1, x1, gz, shape, levels, color='k', interp=True)
mpl.m2km()
mpl.show()
