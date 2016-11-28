
"""
Seismic: 2D straight-ray tomography using smoothness regularization
"""
import numpy as np
from fatiando.mesher import SquareMesh
from fatiando.seismic import ttime2d, srtomo
from fatiando.inversion import Smoothness2D
from fatiando.vis import mpl
from fatiando import utils, gridder

area = (0, 500000, 0, 500000)
shape = (30, 30)
model = SquareMesh(area, shape)
vel = 4000 * np.ones(shape)
vel[5:25, 5:25] = 10000
model.addprop('vp', vel.ravel())

# Make some travel time data and add noise
seed = 0  # Set the random seed so that points are the same every time
src_loc_x, src_loc_y = gridder.scatter(area, 80, seed=seed)
src_loc = np.transpose([src_loc_x, src_loc_y])
rec_loc_x, rec_loc_y = gridder.circular_scatter(area, 30,
                                                random=True, seed=seed)
rec_loc = np.transpose([rec_loc_x, rec_loc_y])
srcs = [src for src in src_loc for _ in rec_loc]
recs = [rec for _ in src_loc for rec in rec_loc]
tts = ttime2d.straight(model, 'vp', srcs, recs)
tts, error = utils.contaminate(tts, 0.02, percent=True, return_stddev=True,
                               seed=seed)
# Make the mesh
mesh = SquareMesh(area, shape)
# and run the inversion
tomo = (srtomo.SRTomo(tts, srcs, recs, mesh) +
        1e8*Smoothness2D(mesh.shape))
tomo.fit()
mesh.addprop('vp', tomo.estimate_)

# Calculate and print the standard deviation of the residuals
# Should be close to the data error if the inversion was able to fit the data
residuals = tomo[0].residuals()
print "Assumed error: %g" % (error)
print "Standard deviation of residuals: %g" % (np.std(residuals))

mpl.figure(figsize=(14, 5))
mpl.subplot(1, 2, 1)
mpl.axis('scaled')
mpl.title('Vp model')
mpl.squaremesh(model, prop='vp', cmap=mpl.cm.seismic)
cb = mpl.colorbar()
cb.set_label('Velocity')
mpl.points(src_loc, '*y', label="Sources")
mpl.points(rec_loc, '^r', label="Receivers")
mpl.legend(loc='lower left', shadow=True, numpoints=1, prop={'size': 10})
mpl.m2km()
mpl.subplot(1, 2, 2)
mpl.axis('scaled')
mpl.title('Tomography result')
mpl.squaremesh(mesh, prop='vp', vmin=4000, vmax=10000,
               cmap=mpl.cm.seismic)
cb = mpl.colorbar()
cb.set_label('Velocity')
mpl.m2km()
mpl.figure()
mpl.grid()
mpl.title('Residuals (data with %.4f s error)' % (error))
mpl.hist(residuals, color='gray', bins=10)
mpl.xlabel("seconds")
mpl.show()
