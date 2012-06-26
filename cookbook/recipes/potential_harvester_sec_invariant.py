"""
Example of inverting the second invariant of the gradient tensor from a complex
source using harvester
"""
import fatiando as ft
import numpy

log = ft.log.get()
log.info(ft.log.header())

# Create a synthetic model
bounds = [-10000, 10000, -10000, 10000, 0, 10000]
area = bounds[0:4]
model = []
for i in xrange(1):
    fig = ft.vis.figure()
    ft.vis.axis('scaled')
    for p in model:
        ft.vis.polygon(p, '.-k', xy2ne=True)
    ft.vis.set_area(area)
    model.append(
        ft.msh.ddd.PolygonalPrism(
            ft.ui.picker.draw_polygon(area, fig.gca(), xy2ne=True),
            1000, 6000, {'density':1000}))
# and generate synthetic data from it
shape = (25, 25)
xp, yp, zp = ft.grd.regular(area, shape, z=-1)
noise = 1 # 1 Eotvos noise
tensor = [ft.utils.contaminate(ft.pot.polyprism.gxx(xp, yp, zp, model), noise),
          ft.utils.contaminate(ft.pot.polyprism.gxy(xp, yp, zp, model), noise),
          ft.utils.contaminate(ft.pot.polyprism.gxz(xp, yp, zp, model), noise),
          ft.utils.contaminate(ft.pot.polyprism.gyy(xp, yp, zp, model), noise),
          ft.utils.contaminate(ft.pot.polyprism.gyz(xp, yp, zp, model), noise),
          ft.utils.contaminate(ft.pot.polyprism.gzz(xp, yp, zp, model), noise)]
inv1, inv2, inv = ft.pot.tensor.invariants(tensor)
# Plot the data and pick the seeds
ft.vis.figure()
ft.vis.suptitle("Pick the seeds (polygon is the true source)")
ft.vis.axis('scaled')
levels = ft.vis.contourf(yp, xp, inv2, shape, 12)
ft.vis.colorbar()
ft.vis.polygon(model[0], xy2ne=True)
ft.vis.xlabel('Horizontal coordinate y (km)')
ft.vis.ylabel('Horizontal coordinate x (km)')
seedx, seedy = ft.ui.picker.points(area, ft.vis.gca(), xy2ne=True).T
spoints = numpy.transpose([seedx, seedy, 3500*numpy.ones_like(seedx)])
ft.vis.show()
# Create a mesh
mesh = ft.msh.ddd.PrismMesh(bounds, (10, 20, 20))
# Make the data modules
dms = ft.pot.harvester.wrapdata(mesh, xp, yp, zp, inv2=inv2)
# Make the seed and set the compactness regularizing parameter mu
seeds = ft.pot.harvester.sow(spoints, {'density':[1000]*len(spoints)},
    mesh, mu=1, delta=0.00001)
# Run the inversion
estimate, goals, misfits = ft.pot.harvester.harvest(dms, seeds)
# Put the estimated density values in the mesh
mesh.addprop('density', estimate['density'])
# Plot the adjustment and the result
predicted = dms[0].get_predicted()
ft.vis.figure()
ft.vis.title("True: color | Predicted: contour")
ft.vis.axis('scaled')
levels = ft.vis.contourf(yp, xp, inv2, shape, 12)
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
