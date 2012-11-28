"""
Potential: 3D gravity inversion by planting anomalous densities using
``harvester`` (simple example)
"""
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())

# Create a synthetic model
model = [ft.msh.ddd.Prism(250, 750, 250, 750, 200, 700, {'density':1000})]
# and generate synthetic data from it
shape = (25, 25)
bounds = [0, 1000, 0, 1000, 0, 1000]
area = bounds[0:4]
xp, yp, zp = ft.gridder.regular(area, shape, z=-1)
noise = 0.1 # 0.1 mGal noise
gz = ft.utils.contaminate(ft.pot.prism.gz(xp, yp, zp, model), noise)
# plot the data
ft.vis.figure()
ft.vis.title("Synthetic gravity anomaly (mGal)")
ft.vis.axis('scaled')
levels = ft.vis.contourf(yp, xp, gz, shape, 12)
ft.vis.colorbar()
ft.vis.xlabel('Horizontal coordinate y (km)')
ft.vis.ylabel('Horizontal coordinate x (km)')
ft.vis.m2km()
ft.vis.show()
# Create a mesh
mesh = ft.msh.ddd.PrismMesh(bounds, (25, 25, 25))
# Make the data modules
dms = ft.pot.harvester.wrapdata(mesh, xp, yp, zp, gz=gz)
# Make the seed and set the compactness regularizing parameter mu
seeds = ft.pot.harvester.sow([[500, 500, 450, {'density':1000}]],
    mesh, mu=0.01, delta=0.0001)
# Run the inversion
estimate, goals, misfits = ft.pot.harvester.harvest(dms, seeds)
# Put the estimated density values in the mesh
mesh.addprop('density', estimate['density'])
# Plot the adjustment and the result
predicted = dms[0].get_predicted()
ft.vis.figure()
ft.vis.title("True: color | Inversion: contour")
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
ft.vis.prisms(model, 'density', style='wireframe')
ft.vis.prisms(ft.msh.ddd.vremove(0, 'density', mesh), 'density')
ft.vis.axes3d(ft.vis.outline3d(bounds),
              ranges=[i*0.001 for i in bounds], fmt='%.1f', nlabels=6)
ft.vis.wall_bottom(bounds)
ft.vis.wall_north(bounds)
ft.vis.show3d()
