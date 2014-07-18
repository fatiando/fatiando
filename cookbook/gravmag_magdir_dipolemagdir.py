"""
GravMag: Use the DipoleMagDir class to estimate the magnetization direction
of dipoles with known centers
"""
import numpy

from fatiando import mesher, gridder
from fatiando.utils import ang2vec, vec2ang, contaminate
from fatiando.gravmag import sphere
from fatiando.vis import mpl
from fatiando.gravmag.magdir import DipoleMagDir
from fatiando.constants import CM

# Make noise-corrupted synthetic data
inc, dec = -10.0, -15.0  # inclination and declination of the Geomagnetic Field
model = [mesher.Sphere(3000, 3000, 1000, 1000,
                       {'magnetization': ang2vec(6.0, -20.0, -10.0)}),
         mesher.Sphere(7000, 7000, 1000, 1000,
                       {'magnetization': ang2vec(10.0, 3.0, -67.0)})]
area = (0, 10000, 0, 10000)
x, y, z = gridder.scatter(area, 1000, z=-150, seed=0)
tf = contaminate(sphere.tf(x, y, z, model, inc, dec), 5.0, seed=0)

# Give the centers of the dipoles
centers = [[3000, 3000, 1000], [7000, 7000, 1000]]

# Estimate the magnetization vectors
solver = DipoleMagDir(x, y, z, tf, inc, dec, centers).fit()

# Print the estimated and true dipole monents, inclinations and declinations
print 'Estimated magnetization (intensity, inclination, declination)'
for e in solver.estimate_:
    print e

# Plot the fit and the normalized histogram of the residuals
mpl.figure(figsize=(14, 5))
mpl.subplot(1, 2, 1)
mpl.title("Total Field Anomaly (nT)", fontsize=14)
mpl.axis('scaled')
nlevels = mpl.contour(y, x, tf, (50, 50), 15, interp=True, color='r',
                      label='Observed', linewidth=2.0)
mpl.contour(y, x, solver.predicted(), (50, 50), nlevels, interp=True,
            color='b', label='Predicted', style='dashed', linewidth=2.0)
mpl.legend(loc='upper left', shadow=True, prop={'size': 13})
mpl.xlabel('East y (m)', fontsize=14)
mpl.ylabel('North x (m)', fontsize=14)
mpl.subplot(1, 2, 2)
residuals_mean = numpy.mean(solver.residuals())
residuals_std = numpy.std(solver.residuals())
# Each residual is subtracted from the mean and the resulting
# difference is divided by the standard deviation
s = (solver.residuals() - residuals_mean) / residuals_std
mpl.hist(s, bins=21, range=None, normed=True, weights=None,
         cumulative=False, bottom=None, histtype='bar', align='mid',
         orientation='vertical', rwidth=None, log=False,
         color=None, label=None)
mpl.xlim(-4, 4)
mpl.title("mean = %.3f    std = %.3f" % (residuals_mean, residuals_std),
          fontsize=14)
mpl.ylabel("P(z)", fontsize=14)
mpl.xlabel("z", fontsize=14)
mpl.show()
