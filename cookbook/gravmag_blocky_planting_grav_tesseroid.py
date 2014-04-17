"""
GravMag: 3D blocky gravity inversion by planting anomalous densities using
tesseroids
"""
from fatiando import gridder, utils
from fatiando.gravmag import tesseroid
from fatiando.gravmag.blocky import sow, Gravity
from fatiando.mesher import Tesseroid, TesseroidMesh, vremove
from fatiando.vis import mpl, myv

# Create a synthetic model
bounds = [-5, 5, 40, 60, 0, -50000]
model = [Tesseroid(-0.5, 0.5, 45, 55, -10000, -30000, {'density':500})]
# and generate synthetic data from it
shape = (50, 50)
area = bounds[0:4]
lon, lat, height = gridder.regular(area, shape, z=10000)
noise = 0.1 # 0.1 mGal noise
gz = utils.contaminate(tesseroid.gz(lon, lat, height, model), noise)

mpl.figure()
mpl.title("Gravity anomaly")
bm = mpl.basemap(area, 'merc')
levels = mpl.contourf(lon, lat, gz, shape, 12, basemap=bm)
mpl.colorbar()
mpl.show()

# Inversion setup
mesh = TesseroidMesh(bounds, (20, 40, 50))
seeds = sow([[0, 47.5, -15000, {'density':500}]], mesh)
solver = Gravity(lon, lat, height, gz, mesh).config(
    'planting', seeds=seeds, compactness=0.1, threshold=0.0001).fit()
mesh.addprop('density', solver.estimate_)

# Plot the adjustment and the result
mpl.figure()
mpl.subplot(1, 2, 1)
mpl.title("True: color | Predicted: contour")
levels = mpl.contourf(lon, lat, gz, shape, 12, basemap=bm)
mpl.colorbar()
mpl.contour(lon, lat, solver.predicted(), shape, levels, color='k', basemap=bm)
mpl.subplot(1, 2, 2)
mpl.title('Residuals')
mpl.hist(solver.residuals(), bins=20)
mpl.show()
# Plot the result
myv.figure()
myv.tesseroids(model[0].split(1, 10, 1), 'density', opacity=0.4, edges=False)
myv.tesseroids(vremove(0, 'density', mesh), 'density', color=(1, 0, 0),
        linewidth=2)
myv.tesseroids(seeds, 'density')
myv.show()
