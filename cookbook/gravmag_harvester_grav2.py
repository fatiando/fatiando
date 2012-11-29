"""
GravMag: 3D gravity inversion by planting anomalous densities using
``harvester`` (more complex example)
"""
from fatiando import logger, gridder, utils, gravmag, mesher
from fatiando.mesher import PolygonalPrism, PrismMesh
from fatiando.vis import mpl, myv

log = logger.get()
log.info(logger.header())

# Create a synthetic model
bounds = [-10000, 10000, -10000, 10000, 0, 10000]
vertices = [[-4948.97959184, -6714.64019851],
            [-2448.97959184, -3141.43920596],
            [ 2448.97959184,   312.65508685],
            [ 6938.7755102 ,  5394.54094293],
            [ 4846.93877551,  6228.28784119],
            [ 2653.06122449,  3409.4292804 ],
            [-3520.40816327, -1434.24317618],
            [-6632.65306122, -6079.4044665 ]]
model = [PolygonalPrism(vertices, 1000, 4000, {'density':1000})]
# and generate synthetic data from it
shape = (25, 25)
area = bounds[0:4]
xp, yp, zp = gridder.regular(area, shape, z=-1)
noise = 0.1 # 0.1 mGal noise
gz = utils.contaminate(gravmag.polyprism.gz(xp, yp, zp, model), noise)
# Create a mesh
mesh = PrismMesh(bounds, (50, 50, 50))
# Make the data modules
dms = gravmag.harvester.wrapdata(mesh, xp, yp, zp, gz=gz)
# Plot the data and pick the seeds
mpl.figure()
mpl.suptitle("Pick the seeds (polygon is the true source)")
mpl.axis('scaled')
levels = mpl.contourf(yp, xp, gz, shape, 12)
mpl.colorbar()
mpl.polygon(model[0], xy2ne=True)
mpl.xlabel('Horizontal coordinate y (km)')
mpl.ylabel('Horizontal coordinate x (km)')
seedx, seedy = mpl.pick_points(area, mpl.gca(), xy2ne=True).T
rawseeds = [[x, y, 2500, {'density':1000}] for x, y in zip(seedx, seedy)]
mpl.show()
# Make the seed and set the compactness regularizing parameter mu
seeds = gravmag.harvester.sow(rawseeds, mesh, mu=0.1, delta=0.0001)
# Run the inversion
estimate, goals, misfits = gravmag.harvester.harvest(dms, seeds)
# Put the estimated density values in the mesh
mesh.addprop('density', estimate['density'])
# Plot the adjustment and the result
predicted = dms[0].get_predicted()
mpl.figure()
mpl.title("True: color | Predicted: contour")
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
myv.polyprisms(model, 'density', opacity=0.6, linewidth=5)
myv.prisms(mesher.vremove(0, 'density', mesh), 'density')
myv.axes(myv.outline(bounds), ranges=[i*0.001 for i in bounds], fmt='%.1f',
    nlabels=6)
myv.wall_bottom(bounds)
myv.wall_north(bounds)
myv.show()
