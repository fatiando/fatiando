"""
Example of inverting synthetic gz data from a single prism using harvester
"""
from matplotlib import pyplot
from fatiando.mesher.ddd import Prism, PrismMesh, extract, vfilter
from fatiando import potential, logger, vis, gridder, utils

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
gz = utils.contaminate(potential.prism.gz(xp, yp, zp, model), noise)
# Create a mesh
mesh = PrismMesh(bounds, (25, 25, 25))
# Make the data modules
dms = potential.harvester.wrapdata(mesh, xp, yp, zp, gz=gz)
# Make the seed and set the compactness regularizing parameter mu
seeds = potential.harvester.sow_prisms([[500, 500, 450]], {'density':[1000]},
    mesh, mu=0.1, delta=0.00001)
# Run the inversion
estimate, goals, misfits = potential.harvester.harvest(dms, seeds)
# Put the estimated density values in the mesh
mesh.addprop('density', estimate['density'])
# and get only the prisms corresponding to the estimate
body = vfilter(500, 1500, 'density', mesh)
# Plot the adjustment and the result
predicted = dms[0].get_predicted()
pyplot.figure()
pyplot.title("True: color | Inversion: contour")
pyplot.axis('scaled')
levels = vis.map.contourf(yp*0.001, xp*0.001, gz, shape, 12)
pyplot.colorbar()
vis.map.contour(yp*0.001, xp*0.001, predicted, shape, levels, color='k')
pyplot.xlabel('Horizontal coordinate y (km)')
pyplot.ylabel('Horizontal coordinate x (km)')
pyplot.show()
vis.vtk.figure()
vis.vtk.prisms(model, extract('density', model), style='wireframe', vmin=0)
vis.vtk.prisms(body, extract('density', body), vmin=0)
vis.vtk.add_axes(vis.vtk.add_outline(bounds),
    ranges=[i*0.001 for i in bounds], fmt='%.1f', nlabels=6)
vis.vtk.wall_bottom(bounds)
vis.vtk.wall_north(bounds)
vis.vtk.mlab.show()
