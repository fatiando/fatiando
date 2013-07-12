"""
GravMag: 3D magnetic inversion by planting anomalous densities using
``harvester``
"""
from fatiando import logger, gridder, utils
from fatiando import gravmag as gm
from fatiando.mesher import Prism, PrismMesh, vremove
from fatiando.vis import mpl, myv

log = logger.get()
log.info(logger.header())

# The regional field
inc, dec = -10, 25
# Model with induced + remanent magnetization
mag = utils.ang2vec(10, 20, -15)
# Create a synthetic model
model = [Prism(250, 750, 250, 750, 200, 700, {'magnetization':mag})]
# and generate synthetic data from it
shape = (25, 25)
bounds = [0, 1000, 0, 1000, 0, 1000]
area = bounds[0:4]
xp, yp, zp = gridder.regular(area, shape, z=-1)
noise = 5
tf = utils.contaminate(gm.prism.tf(xp, yp, zp, model, inc, dec), noise)
# plot the data
mpl.figure()
mpl.title("Synthetic total field anomaly (nT)")
mpl.axis('scaled')
levels = mpl.contourf(yp, xp, tf, shape, 12)
mpl.colorbar()
mpl.xlabel('Horizontal coordinate y (km)')
mpl.ylabel('Horizontal coordinate x (km)')
mpl.m2km()
mpl.show()

# Inversion setup
# Create a mesh
mesh = PrismMesh(bounds, (25, 25, 25))
# Wrap the data so that harvester can use it
data = [gm.harvester.TotalField(xp, yp, zp, tf, inc, dec)]
# Make the seed
seeds = gm.harvester.sow([[500, 500, 450, {'magnetization':mag}]], mesh)
# Run the inversioin
estimate, predicted = gm.harvester.harvest(data, seeds, mesh,
            compactness=1, threshold=0.0001)
#Put the estimated magnetization vector in the mesh
mesh.addprop('magnetization', estimate['magnetization'])

# Plot the adjustment
mpl.figure()
mpl.title("True: color | Inversion: contour")
mpl.axis('scaled')
levels = mpl.contourf(yp, xp, tf, shape, 12)
mpl.colorbar()
mpl.contour(yp, xp, predicted[0], shape, levels, color='k')
mpl.xlabel('Horizontal coordinate y (km)')
mpl.ylabel('Horizontal coordinate x (km)')
mpl.m2km()
residuals = tf - predicted[0]
mpl.figure()
mpl.title('Residuals: mean=%g stddev=%g' % (residuals.mean(), residuals.std()))
mpl.hist(residuals, bins=10)
mpl.xlabel('Residuals (nT)')
mpl.ylabel('# of')
mpl.show()
# Plot the result
myv.figure()
myv.prisms(model, 'magnetization', style='wireframe')
myv.prisms(vremove(0, 'magnetization', mesh), 'magnetization')
myv.axes(myv.outline(bounds), ranges=[i*0.001 for i in bounds], fmt='%.1f',
            nlabels=6)
myv.wall_bottom(bounds)
myv.wall_north(bounds)
myv.show()
