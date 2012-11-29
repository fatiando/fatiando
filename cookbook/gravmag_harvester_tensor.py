"""
GravMag: 3D gravity gradient inversion by planting anomalous densities using
``harvester`` (with non-targeted sources)
"""
from fatiando import logger, gridder, utils, gravmag, mesher
from fatiando.mesher import Prism, PrismMesh
from fatiando.vis import mpl, myv

log = logger.get()
log.info(logger.header())
log.info(__doc__)

# Generate a synthetic model
bounds = [0, 5000, 0, 5000, 0, 1500]
model = [Prism(500, 4500, 3000, 3500, 200, 700, {'density':1200}),
         Prism(3000, 4500, 1800, 2300, 200, 700, {'density':1200}),
         Prism(500, 1500, 500, 1500, 0, 800, {'density':600}),
         Prism(0, 800, 1800, 2300, 0, 200, {'density':600}),
         Prism(4000, 4800, 100, 900, 0, 300, {'density':600}),
         Prism(0, 2000, 4500, 5000, 0, 200, {'density':600}),
         Prism(3000, 4200, 2500, 2800, 200, 700, {'density':-1000}),
         Prism(300, 2500, 1800, 2700, 500, 1000, {'density':-1000}),
         Prism(4000, 4500, 500, 1500, 400, 1000, {'density':-1000}),
         Prism(1800, 3700, 500, 1500, 300, 1300, {'density':-1000}),
         Prism(500, 4500, 4000, 4500, 400, 1300, {'density':-1000})]
# show it
myv.figure()
myv.prisms(model, 'density')
myv.axes(myv.outline(bounds), ranges=[i*0.001 for i in bounds],
              fmt='%.1f', nlabels=6)
myv.wall_bottom(bounds)
myv.wall_north(bounds)
myv.show()
# and use it to generate some tensor data
shape = (51, 51)
area = bounds[0:4]
noise = 2
x, y, z = gridder.regular(area, shape, z=-150)
gyy = utils.contaminate(gravmag.prism.gyy(x, y, z, model), noise)
gyz = utils.contaminate(gravmag.prism.gyz(x, y, z, model), noise)
gzz = utils.contaminate(gravmag.prism.gzz(x, y, z, model), noise)
# Create a prism mesh
mesh = PrismMesh(bounds, (15, 50, 50))
# Make the data modules
datamods = gravmag.harvester.wrapdata(mesh, x, y, z, gyy=gyy, gyz=gyz, gzz=gzz)
# and the seeds
seeds = gravmag.harvester.sow(
    [( 800, 3250, 600, {'density':1200}),
     (1200, 3250, 600, {'density':1200}),
     (1700, 3250, 600, {'density':1200}),
     (2100, 3250, 600, {'density':1200}),
     (2500, 3250, 600, {'density':1200}),
     (2900, 3250, 600, {'density':1200}),
     (3300, 3250, 600, {'density':1200}),
     (3700, 3250, 600, {'density':1200}),
     (4200, 3250, 600, {'density':1200}),
     (3300, 2050, 600, {'density':1200}),
     (3600, 2050, 600, {'density':1200}),
     (4000, 2050, 600, {'density':1200}),
     (4300, 2050, 600, {'density':1200})], mesh, mu=0.1, delta=0.0001)
# Run the inversion and collect the results
estimate, goals, misfits = gravmag.harvester.harvest(datamods, seeds)
# Insert the estimated density values into the mesh
mesh.addprop('density', estimate['density'])
# and get only the prisms corresponding to our estimate
density_model = mesher.vremove(0, 'density', mesh)
# Get the predicted data from the data modules
tensor = (gyy, gyz, gzz)
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
myv.prisms(density_model, 'density', vmin=0)
myv.axes(myv.outline(bounds), ranges=[i*0.001 for i in bounds], fmt='%.1f',
    nlabels=6)
myv.wall_bottom(bounds)
myv.wall_north(bounds)
myv.show()
