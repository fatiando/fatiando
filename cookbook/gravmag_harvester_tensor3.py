"""
GravMag: 3D gravity gradient inversion by planting anomalous densities using
``harvester`` (dipping example)
"""
from fatiando import gridder, utils
from fatiando.gravmag import prism, harvester
from fatiando.mesher import Prism, PrismMesh, vremove
from fatiando.vis import mpl, myv

# Create a synthetic model
props = {'density':1000}
model = [Prism(400, 600, 300, 500, 200, 400, props),
         Prism(400, 600, 400, 600, 400, 600, props),
         Prism(400, 600, 500, 700, 600, 800, props)]
# and generate synthetic data from it
shape = (51, 51)
bounds = [0, 1000, 0, 1000, 0, 1000]
area = bounds[0:4]
xp, yp, zp = gridder.regular(area, shape, z=-150)
noise = 0.5
gxx = utils.contaminate(prism.gxx(xp, yp, zp, model), noise)
gxy = utils.contaminate(prism.gxy(xp, yp, zp, model), noise)
gxz = utils.contaminate(prism.gxz(xp, yp, zp, model), noise)
gyy = utils.contaminate(prism.gyy(xp, yp, zp, model), noise)
gyz = utils.contaminate(prism.gyz(xp, yp, zp, model), noise)
gzz = utils.contaminate(prism.gzz(xp, yp, zp, model), noise)
tensor = [gxx, gxy, gxz, gyy, gyz, gzz]
titles = ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
# plot the data
mpl.figure()
for i in xrange(len(tensor)):
    mpl.subplot(2, 3, i + 1)
    mpl.title(titles[i])
    mpl.axis('scaled')
    levels = mpl.contourf(yp, xp, tensor[i], shape, 30)
    mpl.colorbar()
    mpl.xlabel('y (km)')
    mpl.ylabel('x (km)')
    mpl.m2km()
mpl.show()

# Inversion setup
# Create a mesh
mesh = PrismMesh(bounds, (30, 30, 30))
# Wrap the data so that harvester can use it
data = [harvester.Gxx(xp, yp, zp, gxx),
        harvester.Gxy(xp, yp, zp, gxy),
        harvester.Gxz(xp, yp, zp, gxz),
        harvester.Gyy(xp, yp, zp, gyy),
        harvester.Gyz(xp, yp, zp, gyz),
        harvester.Gzz(xp, yp, zp, gzz)]
# Make the seeds
seeds = harvester.sow([
    [500, 400, 210, {'density':1000}],
    [500, 550, 510, {'density':1000}]], mesh)
# Run the inversioin
estimate, predicted = harvester.harvest(data, seeds, mesh,
    compactness=0.5, threshold=0.001)
# Put the estimated density values in the mesh
mesh.addprop('density', estimate['density'])

# Plot the adjustment
mpl.figure()
mpl.suptitle("True: color | Inversion: contour")
for i in xrange(len(tensor)):
    mpl.subplot(2, 3, i + 1)
    mpl.title(titles[i])
    mpl.axis('scaled')
    levels = mpl.contourf(yp, xp, tensor[i], shape, 12)
    mpl.colorbar()
    mpl.contour(yp, xp, predicted[i], shape, levels, color='k')
    mpl.xlabel('y (km)')
    mpl.ylabel('x (km)')
    mpl.m2km()
mpl.figure()
mpl.suptitle("Residuals")
for i in xrange(len(tensor)):
    residuals = tensor[i] - predicted[i]
    mpl.subplot(2, 3, i + 1)
    mpl.title(titles[i] + ': stddev=%g' % (residuals.std()))
    mpl.hist(residuals, bins=10, color='gray')
    mpl.xlabel('Residuals (Eotvos)')
mpl.show()
# Plot the result
myv.figure()
myv.prisms(model, 'density', style='wireframe')
myv.prisms(seeds, 'density')
myv.axes(myv.outline(bounds), ranges=[i*0.001 for i in bounds], fmt='%.1f',
    nlabels=6)
myv.wall_bottom(bounds)
myv.wall_north(bounds)
myv.figure()
myv.prisms(model, 'density', style='wireframe')
myv.prisms(vremove(0, 'density', mesh), 'density')
myv.axes(myv.outline(bounds), ranges=[i*0.001 for i in bounds], fmt='%.1f',
    nlabels=6)
myv.wall_bottom(bounds)
myv.wall_north(bounds)
myv.show()
