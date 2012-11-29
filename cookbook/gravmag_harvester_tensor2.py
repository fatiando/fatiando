"""
GravMag: 3D gravity gradient inversion by planting anomalous densities using
``harvester`` (all targeted sources)
"""
from fatiando import logger, gridder, utils, gravmag, mesher
from fatiando.mesher import Prism, PrismMesh
from fatiando.vis import mpl, myv

log = logger.get()
log.info(logger.header())
log.info(__doc__)

# Generate a synthetic model
bounds = [0, 5000, 0, 5000, -500, 2000]
model = [Prism(600, 1200, 200, 4200, 400, 900, {'density':1500}),
         Prism(3000, 4000, 1000, 2000, 200, 800, {'density':1000}),
         Prism(2700, 3200, 3700, 4200, 0, 900, {'density':800})]
# show it
myv.figure()
myv.prisms(model, 'density')
myv.axes(myv.outline(bounds), ranges=[i*0.001 for i in bounds],
              fmt='%.1f', nlabels=6)
myv.wall_bottom(bounds)
myv.wall_north(bounds)
myv.show()
# and use it to generate some tensor data
shape = (25, 25)
area = bounds[0:4]
x, y, z = gridder.regular(area, shape, z=-650)
gxy = utils.contaminate(gravmag.prism.gxy(x, y, z, model), 1)
gzz = utils.contaminate(gravmag.prism.gzz(x, y, z, model), 1)
# Create a prism mesh
mesh = PrismMesh(bounds, (20, 50, 50))
# Make the data modules
datamods = gravmag.harvester.wrapdata(mesh, x, y, z, gxy=gxy, gzz=gzz)
# and the seeds
seeds = gravmag.harvester.sow(
    [(901, 701, 750, {'density':1500}),
     (901, 1201, 750, {'density':1500}),
     (901, 1701, 750, {'density':1500}),
     (901, 2201, 750, {'density':1500}),
     (901, 2701, 750, {'density':1500}),
     (901, 3201, 750, {'density':1500}),
     (901, 3701, 750, {'density':1500}),
     (3701, 1201, 501, {'density':1000}),
     (3201, 1201, 501, {'density':1000}),
     (3701, 1701, 501, {'density':1000}),
     (3201, 1701, 501, {'density':1000}),
     (2951, 3951, 301, {'density':800}),
     (2951, 3951, 701, {'density':800})], mesh, mu=10, delta=0.0005)
# Run the inversion and collect the results
estimate, goals, misfits = gravmag.harvester.harvest(datamods, seeds)
# Insert the estimated density values into the mesh
mesh.addprop('density', estimate['density'])
# and get only the prisms corresponding to our estimate
density_model = mesher.vremove(0, 'density', mesh)
# Get the predicted data from the data modules
tensor = (gxy, gzz)
predicted = [dm.get_predicted() for dm in datamods]
# Plot the results
for true, pred in zip(tensor, predicted):
    mpl.figure()
    mpl.title("True: color | Inversion: contour")
    mpl.axis('scaled')
    levels = mpl.contourf(y*0.001, x*0.001, true, shape, 12)
    mpl.colorbar()
    mpl.contour(y*0.001, x*0.001, pred, shape, levels, color='k')
    mpl.xlabel('Horizontal coordinate y (km)')
    mpl.ylabel('Horizontal coordinate x (km)')
mpl.show()
myv.figure()
myv.prisms(model, 'density', style='wireframe')
myv.prisms(density_model, 'density')
myv.axes(myv.outline(bounds), ranges=[i*0.001 for i in bounds], fmt='%.1f',
    nlabels=6)
myv.wall_bottom(bounds)
myv.wall_north(bounds)
myv.show()
