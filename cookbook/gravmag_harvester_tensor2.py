"""
GravMag: 3D gravity gradient inversion by planting anomalous densities using
``harvester`` (3 sources sources)
"""
from fatiando import gridder, utils
from fatiando.gravmag import prism, harvester
from fatiando.mesher import Prism, PrismMesh, vremove
from fatiando.vis import mpl, myv

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
gxy = utils.contaminate(prism.gxy(x, y, z, model), 1)
gzz = utils.contaminate(prism.gzz(x, y, z, model), 1)

# Wrap the data so that harvester can use it
data = [harvester.Gxy(x, y, z, gxy),
        harvester.Gzz(x, y, z, gzz)]
# Create a prism mesh
mesh = PrismMesh(bounds, (20, 50, 50))
# and the seeds
seeds = harvester.sow(
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
     (2951, 3951, 701, {'density':800})],
    mesh)
# Run the inversion and collect the results
estimate, predicted = harvester.harvest(data, seeds, mesh,
    compactness=1, threshold=0.0001)
# Insert the estimated density values into the mesh
mesh.addprop('density', estimate['density'])
# and get only the prisms corresponding to our estimate
density_model = vremove(0, 'density', mesh)

# Plot the results
tensor = (gxy, gzz)
titles = ('gxy', 'gzz')
mpl.figure()
mpl.suptitle("True: color | Inversion: contour")
for i in xrange(len(tensor)):
    mpl.subplot(2, 2, i + 1)
    mpl.title(titles[i])
    mpl.axis('scaled')
    levels = mpl.contourf(y*0.001, x*0.001, tensor[i], shape, 12)
    mpl.colorbar()
    mpl.contour(y*0.001, x*0.001, predicted[i], shape, levels, color='k')
for i in xrange(len(tensor)):
    mpl.subplot(2, 2, i + 3)
    residuals = tensor[i] - predicted[i]
    mpl.title('residuals stddev = %.2f' % (residuals.std()))
    mpl.hist(residuals, bins=10)
    mpl.xlabel('Residual (Eotvos)')
mpl.show()
myv.figure()
myv.prisms(model, 'density', style='wireframe')
myv.prisms(seeds, 'density')
myv.axes(myv.outline(bounds), ranges=[i*0.001 for i in bounds], fmt='%.1f',
    nlabels=6)
myv.wall_bottom(bounds)
myv.wall_north(bounds)
myv.figure()
myv.prisms(model, 'density', style='wireframe')
myv.prisms(density_model, 'density')
myv.axes(myv.outline(bounds), ranges=[i*0.001 for i in bounds], fmt='%.1f',
    nlabels=6)
myv.wall_bottom(bounds)
myv.wall_north(bounds)
myv.show()
