from matplotlib import pyplot
from fatiando import potential, vis, logger, utils, gridder
from fatiando.mesher.volume import Prism3D, PrismMesh3D, extract, vfilter, center
from fatiando.inversion import harvester

log = logger.get()

log.info("Generate synthetic model and data:")
extent = (0, 10000, 0, 10000, 0, 10000)
model = [Prism3D(3500, 6500, 1500, 8500, 1500, 6500, props={'density':800})]

#vis.mayavi_figure()
#vis.prisms3D(model, extract('density', model), vmin=0)
#vis.add_axes3d(vis.add_outline3d(extent=extent))
#vis.wall_bottom(extent)
#vis.wall_north(extent)
#vis.mlab.show()

shape = (50, 50)
x, y, z = gridder.scatter(extent[0:4], 200, z=-1)
gz = utils.contaminate(potential.prism.gz(x, y, z, model), 0.1)

#pyplot.figure()
#pyplot.axis('scaled')
#pyplot.title('Synthetic gz data')
#levels = vis.contourf(y, x, gz, shape, 10, interpolate=True)
#vis.contour(y, x, gz, shape, levels, interpolate=True)
#pyplot.plot(y, x, 'xk')
#pyplot.xlabel('East (km)')
#pyplot.ylabel('North (km)')
#pyplot.show()

log.info("Harvest the results:")
mesh = PrismMesh3D(extent, (10, 10, 10))
rawseeds = [((5000, 5000, 2500), {'density':800})]
seeds = harvester.sow(mesh, rawseeds)

#vis.mayavi_figure()
#vis.prisms3D(model, extract('density', model), opacity=0.3, vmin=0)
#seedmesh = (mesh[s] for s in extract('index', seeds))
#seedprops = (p['density'] for p in extract('props', seeds))
#vis.prisms3D(seedmesh, seedprops, vmin=0)
#vis.add_axes3d(vis.add_outline3d(extent=extent))
#vis.wall_bottom(extent)
#vis.wall_north(extent)
#vis.mlab.show()

gzmod = harvester.PrismGzModule(x, y, z, gz)
regul = harvester.ConcentrationRegularizer(seeds, mesh, 5*10.**(-2), 3.)
jury = harvester.shape_jury(regul, thresh=0.001, maxcmp=4)

results, goals = harvester.harvest(seeds, mesh, [gzmod], jury)
estimate = results['estimate']
for prop in estimate:
    mesh.addprop(prop, estimate[prop])
density_model = vfilter(1, 2000, 'density', mesh)

#pyplot.figure(figsize=(14,8))
#pyplot.subplot(2,2,1)
#pyplot.title("Adjustment")
#pyplot.axis('scaled')
#levels = vis.contourf(y, x, gz, shape, 10, interpolate=True)
#pyplot.colorbar()
#vis.contour(y, x, gzmod.predicted, shape, levels, interpolate=True)
#pyplot.xlabel('East (km)')
#pyplot.ylabel('North (km)')
#pyplot.subplot(2,2,2)
#pyplot.title("Residuals")
#pyplot.axis('scaled')
#vis.pcolor(y, x, gzmod.residuals(gzmod.predicted), shape, interpolate=True)
#pyplot.colorbar()
#pyplot.xlabel('East (km)')
#pyplot.ylabel('North (km)')
#pyplot.subplot(2,1,2)
#pyplot.title("Goal function X iteration")
#pyplot.plot(goals, '.-k')
#pyplot.show()

vis.mayavi_figure()
vis.prisms3D(model, extract('density', model), style='wireframe')
vis.prisms3D(density_model, extract('density', density_model), vmin=0)
outline = vis.add_outline3d(extent=extent)
vis.add_axes3d(outline)
vis.wall_bottom(extent)
vis.wall_north(extent)
vis.mlab.show()

# SECOND ITERATION
log.info("\nSECOND ITERATION:")
mesh = PrismMesh3D(extent, (30, 30, 30))
rawseeds = [(center(c), {'density':800}) for c in density_model]
seeds = harvester.sow(mesh, rawseeds)

vis.mayavi_figure()
vis.prisms3D(model, extract('density', model), opacity=0.3, vmin=0)
seedmesh = (mesh[s] for s in extract('index', seeds))
seedprops = (p['density'] for p in extract('props', seeds))
vis.prisms3D(seedmesh, seedprops, vmin=0)
vis.add_axes3d(vis.add_outline3d(extent=extent))
vis.wall_bottom(extent)
vis.wall_north(extent)
vis.mlab.show()

gzmod = harvester.PrismGzModule(x, y, z, gz)
regul = harvester.ConcentrationRegularizer(seeds, mesh, 1*10.**(-1), 3.)
jury = harvester.shape_jury(regul, thresh=0.00001, maxcmp=10)

results, goals = harvester.harvest(seeds, mesh, [gzmod], jury)
estimate = results['estimate']
for prop in estimate:
    mesh.addprop(prop, estimate[prop])
density_model = vfilter(1, 2000, 'density', mesh)

pyplot.figure(figsize=(14,8))
pyplot.subplot(2,2,1)
pyplot.title("Adjustment")
pyplot.axis('scaled')
levels = vis.contourf(y, x, gz, shape, 10, interpolate=True)
pyplot.colorbar()
vis.contour(y, x, gzmod.predicted, shape, levels, interpolate=True)
pyplot.xlabel('East (km)')
pyplot.ylabel('North (km)')
pyplot.subplot(2,2,2)
pyplot.title("Residuals")
pyplot.axis('scaled')
vis.pcolor(y, x, gzmod.residuals(gzmod.predicted), shape, interpolate=True)
pyplot.colorbar()
pyplot.xlabel('East (km)')
pyplot.ylabel('North (km)')
pyplot.subplot(2,1,2)
pyplot.title("Goal function X iteration")
pyplot.plot(goals, '.-k')
pyplot.show()

vis.mayavi_figure()
vis.prisms3D(model, extract('density', model), style='wireframe')
vis.prisms3D(density_model, extract('density', density_model), vmin=0)
outline = vis.add_outline3d(extent=extent)
vis.add_axes3d(outline)
vis.wall_bottom(extent)
vis.wall_north(extent)
vis.mlab.show()
