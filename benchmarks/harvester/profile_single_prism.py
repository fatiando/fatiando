"""
Example of inverting synthetic gz data from a single prism using harvester
"""
from fatiando import potential, gridder
from fatiando.mesher.volume import Prism3D, PrismMesh3D
from fatiando.inversion import harvester

extent = (0, 10000, 0, 10000, 0, 6000)
model = [Prism3D(4000, 6000, 2000, 8000, 2000, 4000, props={'density':800})]

shape = (50,50)
x, y, z = gridder.regular(extent[0:4], shape, z=-1)
gz = potential.prism.gz(x, y, z, model)
mesh = PrismMesh3D(extent, (15, 25, 25))
rawseeds = [((5000, 5000, 3000), {'density':800})]
seeds = harvester.sow(mesh, rawseeds)    
gzmod = harvester.PrismGzModule(x, y, z, gz)
jury = harvester.shape_jury(None, thresh=0.0001, tol=0.1, compact=3)
import cProfile
cProfile.run('harvester.harvest(seeds, mesh, [gzmod], jury)')
