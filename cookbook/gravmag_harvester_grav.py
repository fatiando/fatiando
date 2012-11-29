"""
GravMag: 3D gravity inversion by planting anomalous densities using
``harvester`` (simple example)
"""
from fatiando import logger, gridder, utils, gravmag, mesher
from fatiando.mesher import Prism, PrismMesh
from fatiando.vis import mpl, myv

log = logger.get()
log.info(logger.header())

# Create a synthetic model
model = [Prism(250, 750, 250, 750, 200, 700, {'density':1000})]
# and generate synthetic data from it
shape = (25, 25)
bounds = [0, 1000, 0, 1000, 0, 1000]
area = bounds[0:4]
xp, yp, zp = gridder.regular(area, shape, z=-1)
noise = 0.1 # 0.1 mGal noise
gz = utils.contaminate(gravmag.prism.gz(xp, yp, zp, model), noise)
# plot the data
mpl.figure()
mpl.title("Synthetic gravity anomaly (mGal)")
mpl.axis('scaled')
levels = mpl.contourf(yp, xp, gz, shape, 12)
mpl.colorbar()
mpl.xlabel('Horizontal coordinate y (km)')
mpl.ylabel('Horizontal coordinate x (km)')
mpl.m2km()
mpl.show()
# Create a mesh
mesh = PrismMesh(bounds, (25, 25, 25))
# Make the data modules
dms = gravmag.harvester.wrapdata(mesh, xp, yp, zp, gz=gz)
# Make the seed and set the compactness regularizing parameter mu
seeds = gravmag.harvester.sow([[500, 500, 450, {'density':1000}]],
    mesh, mu=0.01, delta=0.0001)
# Run the inversion
estimate, goals, misfits = gravmag.harvester.harvest(dms, seeds)
# Put the estimated density values in the mesh
mesh.addprop('density', estimate['density'])
# Plot the adjustment and the result
predicted = dms[0].get_predicted()
mpl.figure()
mpl.title("True: color | Inversion: contour")
mpl.axis('scaled')
levels = mpl.contourf(yp, xp, gz, shape, 12)
mpl.colorbar()
mpl.contour(yp, xp, predicted, shape, levels, color='k')
mpl.xlabel('Horizontal coordinate y (km)')
mpl.ylabel('Horizontal coordinate x (km)')
mpl.m2km()
mpl.show()
# Plot the result
myv.figure()
myv.prisms(model, 'density', style='wireframe')
myv.prisms(mesher.vremove(0, 'density', mesh), 'density')
myv.axes(myv.outline(bounds), ranges=[i*0.001 for i in bounds], fmt='%.1f',
    nlabels=6)
myv.wall_bottom(bounds)
myv.wall_north(bounds)
myv.show()
