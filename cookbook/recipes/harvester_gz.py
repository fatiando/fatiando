"""
Example of inverting synthetic gz data from a single prism using harvester
"""
from matplotlib import pyplot
from fatiando import potential, vis, logger, utils, gridder
from fatiando.mesher.ddd import Prism, PrismMesh, extract, vfilter
from fatiando.potential import harvester

log = logger.get()
log.info(__doc__)

log.info("First make the synthetic model:\n")
extent = (0, 6000, 0, 6000, 0, 6000)
model = [Prism(2500, 3500, 1000, 5000, 1000, 2000, props={'density':800})]

#vis.mayavi_figure()
#vis.prisms3D(model, extract('density', model), vmin=0)
#vis.add_axes3d(vis.add_outline3d(extent=extent))
#vis.wall_bottom(extent)
#vis.wall_north(extent)
#vis.mlab.show()

log.info("\nSecond calculate the synthetic data:")
shape = (25, 25)
#x, y, z = gridder.scatter(extent[0:4], 200, z=-1)
x, y, z = gridder.regular(extent[0:4], shape, z=-1)
gz = utils.contaminate(potential.prism.gz(x, y, z, model), 0.1)
#gxx = utils.contaminate(potential.prism.gxx(x, y, z, model), 2)

#pyplot.figure()
#pyplot.axis('scaled')
#pyplot.title('Synthetic data')
#levels = vis.contourf(y, x, gzz, shape, 10)
#vis.contour(y, x, gzz, shape, levels)
#pyplot.plot(y, x, 'xk')
#pyplot.xlabel('East (km)')
#pyplot.ylabel('North (km)')
#pyplot.show()

log.info("\nThird make a prism mesh:")
#mesh = PrismMesh3D(extent, (60, 60, 60))
mesh = PrismMesh(extent, (30, 30, 30))
#mesh = PrismMesh(extent, (15, 15, 15))

log.info("\nFourth sow the seeds:")
rawseeds = [((3000, 2000, 1000), {'density':800})]
#rawseeds = [((3000, 2000, 1100), {'density':800}),
            #((3000, 4000, 1100), {'density':800})]
seeds = harvester.sow(mesh, rawseeds)

#vis.mayavi_figure()
#vis.prisms3D(model, extract('density', model), opacity=0.3, vmin=0)
#seedmesh = (mesh[int(s)] for s in extract('index', seeds))
#seedprops = (p['density'] for p in extract('props', seeds))
#vis.prisms3D(seedmesh, seedprops, vmiorn=0)
#vis.add_axes3d(vis.add_outline3d(extent=extent))
#vis.wall_bottom(extent)
#vis.wall_north(extent)
#vis.mlab.show()
    
log.info("\nFith harvest the results:")
datamods = [harvester.PrismGzModule(x, y, z, gz)]
#datamods = [harvester.PrismGzzModule(x, y, z, gzz),
#harvester.PrismGxxModule(x, y, z, gxx)]
#regul = harvester.ConcentrationRegularizer(seeds, mesh, 1*10.**(10), 5.)
#jury = harvester.standard_jury(regul, thresh=0.0001)
regul = harvester.ConcentrationRegularizer(seeds, mesh, 2*10.**(-3), 1.)
jury = harvester.shape_jury(regul, thresh=0.001, maxcmp=4)

results, goals = harvester.harvest(seeds, mesh, datamods, jury)
estimate = results['estimate']
for prop in estimate:
    mesh.addprop(prop, estimate[prop])
density_model = vfilter(1, 2000, 'density', mesh)

import numpy

#goals = []
#for chset in harvester.grow(seeds, mesh, datamods, jury):    
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

for dm in datamods:
    pyplot.figure(figsize=(14,8))
    pyplot.subplot(1,2,1)
    pyplot.title("Adjustment")
    pyplot.axis('scaled')
    levels = vis.map.contour(y, x, dm.obs, shape, 8, color='b')
    vis.map.contour(y, x, dm.predicted, shape, levels, color='r')
    pyplot.xlabel('East (km)')
    pyplot.ylabel('North (km)')
    pyplot.subplot(1,2,2)
    pyplot.title("Residuals")
    pyplot.axis('scaled')
    vis.map.pcolor(y, x, dm.residuals(dm.predicted), shape)
    pyplot.colorbar()
    pyplot.xlabel('East (km)')
    pyplot.ylabel('North (km)')
pyplot.figure()
pyplot.title("Goal function X iteration")
pyplot.plot(goals, '.-k')
pyplot.plot(regul.timeline, '.-g')
pyplot.plot(numpy.array(goals) - numpy.array(regul.timeline), '.-r')
pyplot.show()

vis.vtk.figure()
vis.vtk.prisms(model, extract('density', model), style='wireframe')
vis.vtk.prisms(density_model, extract('density', density_model), vmin=0)
outline = vis.vtk.add_outline(extent=extent)
vis.vtk.add_axes(outline)
vis.vtk.wall_bottom(extent)
vis.vtk.wall_north(extent)
vis.vtk.mlab.show()    
