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
bounds = [-2, 2, 40, 60, 0, -15000]
model = [Tesseroid(-0.1, 0.1, 45, 55, -1000, -9000, {'density':300})]
# and generate synthetic data from it
shape = (50, 50)
area = [-10, 10, 40, 60]
lon, lat, height = gridder.regular(area, shape, z=10000)
noise = 0.1 # 0.1 mGal noise
gz = utils.contaminate(tesseroid.gz(lon, lat, height, model), noise)

mpl.figure()
mpl.title("Gravity anomaly")
bm = mpl.basemap(area, 'merc')
mpl.contourf(lon, lat, gz, shape, 12, basemap=bm)
mpl.colorbar().set_label('mGal')
mpl.draw_geolines(area, 5, 5, basemap=bm)
mpl.show()

# Inversion setup
mesh = TesseroidMesh(bounds, (15, 40, 80))
seeds = sow([[0, 47.5, -1000, {'density':300}]], mesh)
solver = Gravity(lon, lat, height, gz, mesh).config(
    'planting', seeds=seeds, compactness=0.1, threshold=0.0001).fit()
mesh.addprop('density', solver.estimate_)

# Plot the adjustment and the result
mpl.figure()
mpl.subplot(1, 3, 1)
mpl.title("Observed")
mpl.contourf(lon, lat, gz, shape, 12, basemap=bm)
mpl.colorbar().set_label('mgal')
mpl.draw_geolines(area, 5, 5, basemap=bm)
mpl.subplot(1, 3, 2)
mpl.title("Predicted")
mpl.contourf(lon, lat, solver.predicted(), shape, 12, basemap=bm)
mpl.colorbar().set_label('mgal')
mpl.draw_geolines(area, 5, 5, basemap=bm)
mpl.subplot(1, 3, 3)
mpl.title('Residuals')
mpl.hist(solver.residuals(), bins=20)
mpl.show()
# Plot the result
myv.figure()
scale = (5, 1, 5)
myv.tesseroids(model[0].split(1, 3, 1), 'density', opacity=0.4, edges=False,
               scale=scale)
myv.tesseroids(vremove(0, 'density', mesh), 'density', color=(1, 0, 0),
               scale=scale, linewidth=2)
myv.tesseroids(seeds, 'density', scale=scale)
myv.show()
