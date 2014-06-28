"""
GravMag: Using data weights in 3D inversion using :mod:`harvester`
"""
from fatiando import utils, gridder, mesher
from fatiando.gravmag import prism, harvester
from fatiando.vis import mpl, myv

# Generate some synthetic total field anomaly data
bounds = [0, 10000, 0, 10000, 0, 5000]
props = {'density': 500}
props2 = {'density': 1000}
model = [mesher.Prism(4000, 6000, 4000, 6000, 500, 2500, props),
         mesher.Prism(2000, 2500, 2000, 2500, 500, 1000, props2),
         mesher.Prism(7500, 8000, 5500, 6500, 500, 1000, props2),
         mesher.Prism(1500, 2000, 4000, 5000, 500, 1000, props2)]
area = bounds[:4]
shape = (50, 50)
x, y, z = gridder.regular(area, shape, z=-1)
gz = utils.contaminate(prism.gz(x, y, z, model), 0.1)
mesh = mesher.PrismMesh(bounds, (20, 40, 40))
seeds = harvester.sow([[5000, 5000, 1000, props]], mesh)

# Run the inversion without using weights
data = [harvester.Gz(x, y, z, gz)]
estimate, predicted = harvester.harvest(data, seeds, mesh,
                                        compactness=1.5, threshold=0.001)
mesh.addprop('density', estimate['density'])
bodies = mesher.vremove(0, 'density', mesh)
mpl.figure()
mpl.axis('scaled')
mpl.title('No weights: Observed (color) vs Predicted (black)')
levels = mpl.contourf(y, x, gz, shape, 17)
mpl.colorbar()
mpl.contour(y, x, predicted[0], shape, levels, color='k')
mpl.m2km()
mpl.xlabel('East (km)')
mpl.ylabel('North (km)')
myv.figure()
plot = myv.prisms(model, 'density', style='wireframe', linewidth=4)
plot.actor.mapper.scalar_visibility = False
myv.prisms(bodies, 'density')
myv.axes(myv.outline(bounds))
myv.wall_north(bounds)
myv.wall_bottom(bounds)
myv.title('No weights')

# Run the inversion again with weights
weights = harvester.weights(x, y, seeds, [2000], decay=6)
data = [harvester.Gz(x, y, z, gz, weights=weights)]
estimate, predicted = harvester.harvest(data, seeds, mesh,
                                        compactness=1.5, threshold=0.001)
mesh.addprop('density', estimate['density'])
bodies = mesher.vremove(0, 'density', mesh)
mpl.figure()
mpl.axis('scaled')
mpl.title('With weights: Observed (color) vs Predicted (black)')
levels = mpl.contourf(y, x, gz, shape, 17)
mpl.colorbar()
mpl.contour(y, x, predicted[0], shape, levels, color='k')
mpl.m2km()
mpl.xlabel('East (km)')
mpl.ylabel('North (km)')
mpl.figure()
mpl.axis('scaled')
mpl.title('Isolated anomaly (color) vs Predicted (black)')
levels = mpl.contourf(y, x, prism.gz(x, y, z, [model[0]]), shape, 15)
mpl.colorbar()
mpl.contour(y, x, predicted[0], shape, levels, color='k')
mpl.m2km()
mpl.xlabel('East (km)')
mpl.ylabel('North (km)')
mpl.show()
myv.figure()
plot = myv.prisms(model, 'density', style='wireframe', linewidth=4)
plot.actor.mapper.scalar_visibility = False
myv.prisms([mesh[s.i] for s in seeds], 'density')
myv.prisms(bodies, 'density')
myv.axes(myv.outline(bounds))
myv.wall_north(bounds)
myv.wall_bottom(bounds)
myv.title('With weights')
myv.show()
