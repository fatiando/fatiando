from matplotlib import pyplot
from fatiando import potential, vis, logger, utils, gridder
from fatiando.mesher.volume import Prism3D, PrismMesh3D, extract, vfilter
from fatiando.inversion import harvester

log = logger.get()

extent = (0, 10000, 0, 10000, 0, 10000)
model = [Prism3D(3500, 6500, 1500, 8500, 1500, 6500, props={'density':800})]

vis.mayavi_figure()
vis.prisms3D(model, extract('density', model), vmin=0)
vis.add_axes3d(vis.add_outline3d(extent=extent))
vis.wall_bottom(extent)
vis.wall_north(extent)
vis.mlab.show()

shape = (50, 50)
x, y, z = gridder.scatter(extent[0:4], 200, z=-1)
gz = utils.contaminate(potential.prism.gz(x, y, z, model), 0.1)

pyplot.figure()
pyplot.axis('scaled')
pyplot.title('Synthetic gz data')
levels = vis.contourf(y, x, gz, shape, 10, interpolate=True)
vis.contour(y, x, gz, shape, levels, interpolate=True)
pyplot.plot(y, x, 'xk')
pyplot.xlabel('East (km)')
pyplot.ylabel('North (km)')
pyplot.show()

log.info("\nThird make a prism mesh:")
#mesh = PrismMesh3D(extent, (25, 25, 25))
mesh = PrismMesh3D(extent, (10, 10, 10))


log.info("\nFourth sow the seeds:")
rawseeds = [((5000, 5000, 2500), {'density':800})]
seeds = harvester.sow(mesh, rawseeds)

vis.mayavi_figure()
vis.prisms3D(model, extract('density', model), opacity=0.3, vmin=0)
seedmesh = (mesh[int(s)] for s in extract('index', seeds))
seedprops = (p['density'] for p in extract('props', seeds))
vis.prisms3D(seedmesh, seedprops, vmin=0)
vis.add_axes3d(vis.add_outline3d(extent=extent))
vis.wall_bottom(extent)
vis.wall_north(extent)
vis.mlab.show()
    
log.info("\nFith harvest the results:")
gzmod = harvester.PrismGzModule(x, y, z, gz)
regul = harvester.ConcentrationRegularizer(seeds, mesh, 1*10.**(1), 3.)
jury = harvester.standard_jury(regul, thresh=0.001)

results, goals = harvester.harvest(seeds, mesh, [gzmod], jury)
estimate = results['estimate']
for prop in estimate:
    mesh.addprop(prop, estimate[prop])
density_model = vfilter(1, 2000, 'density', mesh)

#import numpy
#goals = []
#for chset in harvester.grow(seeds, mesh, [gzmod], jury):    
    #estimate = chset['estimate']
    #goals.append(chset['goal'])
    #for prop in estimate:
        #mesh.addprop(prop, estimate[prop])
    #density_model = vfilter(1, 2000, 'density', mesh)
    #neighbors = [mesh[n['index']] for nei in chset['neighborhood'] for n in nei]
    #vis.mayavi_figure()
    #vis.prisms3D(model, extract('density', model), style='wireframe', vmin=0)
    #vis.prisms3D(neighbors, numpy.zeros_like(neighbors), style='wireframe')
    #vis.prisms3D(density_model, extract('density', density_model), vmin=0)
    #vis.add_axes3d(vis.add_outline3d(extent=extent))
    #vis.mlab.show()

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
