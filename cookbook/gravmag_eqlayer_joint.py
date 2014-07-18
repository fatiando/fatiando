"""
GravMag: Fit an equivalent layer to gravity and gravity gradient data
"""
import numpy as np
from fatiando.gravmag import prism, sphere
from fatiando.gravmag.eqlayer import EQLGravity
from fatiando.inversion.regularization import Smoothness2D, LCurve
from fatiando import gridder, utils, mesher
from fatiando.vis import mpl

# Make synthetic data
props = {'density': 1000}
model = [mesher.Prism(-500, 500, -1000, 1000, 500, 4000, props)]
area = [-5000, 5000, -5000, 5000]
x1, y1, z1 = gridder.scatter(area, 80, z=0, seed=0)
gz = utils.contaminate(prism.gz(x1, y1, z1, model), 0.1, seed=0)
x2, y2, z2 = gridder.regular(area, (10, 50), z=-200)
gzz = utils.contaminate(prism.gzz(x2, y2, z2, model), 5, seed=0)
# Setup the layer
layer = mesher.PointGrid([-6000, 6000, -6000, 6000], 500, (50, 50))
# and the inversion
# Apply a scaling factor to make both portions of the misfit the same order of
# magnitude
scale = np.linalg.norm(gz) ** 2 / np.linalg.norm(gzz) ** 2
misfit = (EQLGravity(x1, y1, z1, gz, layer)
          + scale * EQLGravity(x2, y2, z2, gzz, layer, field='gzz'))
regul = Smoothness2D(layer.shape)
# Use an L-curve analysis to find the best regularization parameter
solver = LCurve(misfit, regul, [10 ** i for i in range(-30, -20)]).fit()
layer.addprop('density', solver.estimate_)

# Now I can forward model gz using my layer to produce an integrated map in a
# much denser region
shape = (50, 50)
x, y, z = gridder.regular(area, shape, z=0)
gz_layer = sphere.gz(x, y, z, layer)
gz_true = prism.gz(x, y, z, model)

mpl.figure()
mpl.suptitle('L-curve')
mpl.title("Estimated regularization parameter: %g" % (solver.regul_param_))
solver.plot_lcurve()
mpl.grid()

# Plot the layer and the fit
mpl.figure(figsize=(14, 4))
mpl.suptitle('Observed data (black) | Predicted by layer (red)')
mpl.subplot(1, 3, 1)
mpl.axis('scaled')
mpl.title('Layer (kg.m^-3)')
mpl.pcolor(layer.y, layer.x, layer.props['density'], layer.shape)
mpl.colorbar().set_label(r'Density $kg.m^{-3}$')
mpl.m2km()
mpl.subplot(1, 3, 2)
mpl.axis('scaled')
mpl.title('Fit gz (mGal)')
levels = mpl.contour(y1, x1, gz, shape, 15, color='k', interp=True)
mpl.contour(y1, x1, solver.predicted()[0], shape, levels, color='r',
            interp=True)
mpl.plot(y1, x1, 'xk', label='Data points')
mpl.legend()
mpl.m2km()
mpl.subplot(1, 3, 3)
mpl.axis('scaled')
mpl.title('Fit gzz (Eotvos)')
levels = mpl.contour(y2, x2, gzz, shape, 10, color='k', interp=True)
mpl.contour(y2, x2, solver.predicted()[1], shape, levels, color='r',
            interp=True)
mpl.plot(y2, x2, 'xk', label='Data points')
mpl.legend()
mpl.m2km()

mpl.figure()
mpl.suptitle('Interpolated gz at ground level')
mpl.subplot(1, 2, 1)
mpl.axis('scaled')
mpl.title('True (black) | Layer (red)')
levels = mpl.contour(y, x, gz_true, shape, 12, color='k')
mpl.contour(y, x, gz_layer, shape, levels, color='r')
mpl.m2km()
mpl.subplot(1, 2, 2)
mpl.axis('scaled')
mpl.title('True (black) | Interpolation (red)')
levels = mpl.contour(y, x, gz_true, shape, 12, color='k')
mpl.contour(y1, x1, gz, shape, levels, color='r', interp=True)
mpl.m2km()
mpl.show()
