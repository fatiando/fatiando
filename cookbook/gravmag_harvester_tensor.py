"""
GravMag: 3D gravity gradient inversion by planting anomalous densities using
``harvester`` (with non-targeted sources)
"""
from fatiando import gridder, utils
from fatiando.gravmag import prism, harvester
from fatiando.mesher import Prism, PrismMesh, vremove
from fatiando.vis import mpl, myv

# Generate a synthetic model
bounds = [0, 5000, 0, 5000, 0, 1500]
model = [Prism(500, 4500, 3000, 3500, 200, 700, {'density': 1200}),
         Prism(3000, 4500, 1800, 2300, 200, 700, {'density': 1200}),
         Prism(500, 1500, 500, 1500, 0, 800, {'density': 600}),
         Prism(0, 800, 1800, 2300, 0, 200, {'density': 600}),
         Prism(4000, 4800, 100, 900, 0, 300, {'density': 600}),
         Prism(0, 2000, 4500, 5000, 0, 200, {'density': 600}),
         Prism(3000, 4200, 2500, 2800, 200, 700, {'density': -1000}),
         Prism(300, 2500, 1800, 2700, 500, 1000, {'density': -1000}),
         Prism(4000, 4500, 500, 1500, 400, 1000, {'density': -1000}),
         Prism(1800, 3700, 500, 1500, 300, 1300, {'density': -1000}),
         Prism(500, 4500, 4000, 4500, 400, 1300, {'density': -1000})]
# show it
myv.figure()
myv.prisms(model, 'density')
myv.axes(myv.outline(bounds), ranges=[i * 0.001 for i in bounds],
         fmt='%.1f', nlabels=6)
myv.wall_bottom(bounds)
myv.wall_north(bounds)
myv.show()
# and use it to generate some tensor data
shape = (51, 51)
area = bounds[0:4]
noise = 2
x, y, z = gridder.regular(area, shape, z=-150)
gyy = utils.contaminate(prism.gyy(x, y, z, model), noise)
gyz = utils.contaminate(prism.gyz(x, y, z, model), noise)
gzz = utils.contaminate(prism.gzz(x, y, z, model), noise)

# Set up the inversion:
# Create a prism mesh
mesh = PrismMesh(bounds, (15, 50, 50))
# Wrap the data so that harvester can use it
data = [harvester.Gyy(x, y, z, gyy),
        harvester.Gyz(x, y, z, gyz),
        harvester.Gzz(x, y, z, gzz)]
# and the seeds
seeds = harvester.sow(
    [(800, 3250, 600, {'density': 1200}),
     (1200, 3250, 600, {'density': 1200}),
     (1700, 3250, 600, {'density': 1200}),
     (2100, 3250, 600, {'density': 1200}),
     (2500, 3250, 600, {'density': 1200}),
     (2900, 3250, 600, {'density': 1200}),
     (3300, 3250, 600, {'density': 1200}),
     (3700, 3250, 600, {'density': 1200}),
     (4200, 3250, 600, {'density': 1200}),
     (3300, 2050, 600, {'density': 1200}),
     (3600, 2050, 600, {'density': 1200}),
     (4000, 2050, 600, {'density': 1200}),
     (4300, 2050, 600, {'density': 1200})],
    mesh)
# Run the inversion and collect the results
estimate, predicted = harvester.harvest(data, seeds, mesh,
                                        compactness=1., threshold=0.0001)

# Insert the estimated density values into the mesh
mesh.addprop('density', estimate['density'])
# and get only the prisms corresponding to our estimate
density_model = vremove(0, 'density', mesh)
print "Accretions: %d" % (len(density_model) - len(seeds))

# Get the predicted data from the data modules
tensor = (gyy, gyz, gzz)
# plot it
for true, pred in zip(tensor, predicted):
    mpl.figure()
    mpl.title("True: color | Inversion: contour")
    mpl.axis('scaled')
    levels = mpl.contourf(y * 0.001, x * 0.001, true, shape, 12)
    mpl.colorbar()
    mpl.contour(y * 0.001, x * 0.001, pred, shape, levels, color='k')
    mpl.xlabel('Horizontal coordinate y (km)')
    mpl.ylabel('Horizontal coordinate x (km)')
mpl.show()

# Plot the inversion result
myv.figure()
myv.prisms(model, 'density', style='wireframe')
myv.prisms(density_model, 'density', vmin=0)
myv.axes(myv.outline(bounds), ranges=[i * 0.001 for i in bounds], fmt='%.1f',
         nlabels=6)
myv.wall_bottom(bounds)
myv.wall_north(bounds)
myv.show()
