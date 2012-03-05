"""
Example of inverting synthetic gz data from a single prism using harvester
"""
from fatiando import potential, gridder, logger
from fatiando.mesher.ddd import Prism, PrismMesh
print logger.header()
model = [Prism(250, 750, 250, 750, 200, 700, {'density':1000})]
shape = (25, 25)
bounds = [0, 1000, 0, 1000, 0, 1000]
area = bounds[0:4]
xp, yp, zp = gridder.regular(area, shape, z=-1)
gz = potential.prism.gz(xp, yp, zp, model)
mesh = PrismMesh(bounds, (25, 25, 25))
dms = potential.harvester.wrapdata(mesh, xp, yp, zp, gz=gz)
seeds = potential.harvester.sow_prisms([[500, 500, 450]], {'density':[1000]},
    mesh, mu=0.1, delta=0.00001)
import cProfile
cProfile.run('potential.harvester.harvest(dms, seeds)')
