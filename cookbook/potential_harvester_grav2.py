"""
Potential: 3D gravity inversion by planting anomalous densities using
``harvester`` (more complex example)
"""
import fatiando as ft
import numpy

log = ft.logger.get()
log.info(ft.logger.header())

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
model = [ft.msh.ddd.PolygonalPrism(vertices, 1000, 4000, {'density':1000})]
# and generate synthetic data from it
shape = (25, 25)
area = bounds[0:4]
xp, yp, zp = ft.gridder.regular(area, shape, z=-1)
noise = 0.1 # 0.1 mGal noise
gz = ft.utils.contaminate(ft.pot.polyprism.gz(xp, yp, zp, model), noise)
# Create a mesh
mesh = ft.msh.ddd.PrismMesh(bounds, (50, 50, 50))
# Make the data modules
dms = ft.pot.harvester.wrapdata(mesh, xp, yp, zp, gz=gz)
# Plot the data and pick the seeds
ft.vis.figure()
ft.vis.suptitle("Pick the seeds (polygon is the true source)")
ft.vis.axis('scaled')
levels = ft.vis.contourf(yp, xp, gz, shape, 12)
ft.vis.colorbar()
ft.vis.polygon(model[0], xy2ne=True)
ft.vis.xlabel('Horizontal coordinate y (km)')
ft.vis.ylabel('Horizontal coordinate x (km)')
seedx, seedy = ft.ui.picker.points(area, ft.vis.gca(), xy2ne=True).T
rawseeds = [[x, y, 2500, {'density':1000}] for x, y in zip(seedx, seedy)]
ft.vis.show()
# Make the seed and set the compactness regularizing parameter mu
seeds = ft.pot.harvester.sow(rawseeds, mesh, mu=0.1, delta=0.0001)
# Run the inversion
estimate, goals, misfits = ft.pot.harvester.harvest(dms, seeds)
# Put the estimated density values in the mesh
mesh.addprop('density', estimate['density'])
# Plot the adjustment and the result
predicted = dms[0].get_predicted()
ft.vis.figure()
ft.vis.title("True: color | Predicted: contour")
ft.vis.axis('scaled')
levels = ft.vis.contourf(yp, xp, gz, shape, 12)
ft.vis.colorbar()
ft.vis.contour(yp, xp, predicted, shape, levels, color='k')
ft.vis.xlabel('Horizontal coordinate y (km)')
ft.vis.ylabel('Horizontal coordinate x (km)')
ft.vis.m2km()
ft.vis.show()
# Plot the result
ft.vis.figure3d()
ft.vis.polyprisms(model, 'density', opacity=0.6, linewidth=5)
ft.vis.prisms(ft.msh.ddd.vremove(0, 'density', mesh), 'density')
ft.vis.axes3d(ft.vis.outline3d(bounds),
              ranges=[i*0.001 for i in bounds], fmt='%.1f', nlabels=6)
ft.vis.wall_bottom(bounds)
ft.vis.wall_north(bounds)
ft.vis.show3d()
