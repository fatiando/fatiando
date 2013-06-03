"""
GravMag: Iterate through a 3D gravity inversion by planting anomalous densities
"""
from fatiando import gridder, gravmag
from fatiando.mesher import Prism, PrismMesh
from fatiando.vis import myv

model = [Prism(200, 800, 400, 600, 200, 400, {'density':1000})]
shape = (20, 20)
bounds = [0, 1000, 0, 1000, 0, 1000]
area = bounds[0:4]
x, y, z = gridder.regular(area, shape, z=-1)
gz = gravmag.prism.gz(x, y, z, model)
mesh = PrismMesh(bounds, (10, 10, 10))
data = [gravmag.harvester.Gz(x, y, z, gz)]
seeds = gravmag.harvester.sow([[500, 500, 250, {'density':1000}]], mesh)

fig = myv.figure(size=(700, 700))
plot = myv.prisms(model, style='wireframe', linewidth=4)
plot.actor.mapper.scalar_visibility = False
myv.prisms([mesh[s.i] for s in seeds])
myv.outline(bounds)
myv.wall_bottom(bounds)
myv.wall_east(bounds)
for update in gravmag.harvester.iharvest(data, seeds, mesh, compactness=0.5,
        threshold=0.001):
    best, neighborhood = update[2:4]
    if best is not None:
        myv.prisms([mesh[best.i]])
        plot = myv.prisms(
            [mesh[n] for neighbors in neighborhood for n in neighbors],
            style='wireframe')
        plot.actor.mapper.scalar_visibility = False
myv.show()
