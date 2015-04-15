"""
GravMag: 2D gravity inversion for the relief of a basin
"""
from fatiando.inversion.regularization import Smoothness1D, LCurve
from fatiando.gravmag.basin2d import PolygonalBasinGravity
from fatiando.gravmag import talwani
from fatiando.mesher import Polygon
from fatiando.vis import mpl
from fatiando import utils
import numpy as np

# Make some synthetic data to test the inversion
# The model will be a polygon.
# Reverse x because vertices must be clockwise.
xs = np.linspace(0, 100000, 100)[::-1]
depths = (-1e-15*(xs - 50000)**4 + 8000 -
          3000*np.exp(-(xs - 70000)**2/(10000**2)))
depths -= depths.min()  # Reduce depths to zero
props = {'density': -300}
model = Polygon(np.transpose([xs, depths]), props)
x = np.linspace(0, 100000, 100)
z = -100*np.ones_like(x)
data = utils.contaminate(talwani.gz(x, z, [model]), 0.5, seed=0)

# Make the solver and run the inversion
misfit = PolygonalBasinGravity(x, z, data, 50, props, top=0)
regul = Smoothness1D(misfit.nparams)
# Use an L-curve analysis to find the best regularization parameter
lc = LCurve(misfit, regul, [10**i for i in np.arange(-10, -5, 0.5)], jobs=4)
initial = 3000*np.ones(misfit.nparams)
lc.config('levmarq', initial=initial).fit()

mpl.figure()
mpl.subplot(2, 2, 1)
mpl.plot(x, data, 'ok', label='observed')
mpl.plot(x, lc.predicted(), '-r', linewidth=2, label='predicted')
mpl.legend()
ax = mpl.subplot(2, 2, 3)
mpl.polygon(model, fill='gray', alpha=0.5)
mpl.polygon(lc.estimate_, style='o-r')
ax.invert_yaxis()
mpl.subplot(1, 2, 2)
mpl.title('L-curve')
lc.plot_lcurve()
mpl.show()
