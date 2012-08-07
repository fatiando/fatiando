"""
Example of inverting synthetic gz data from a single prism using harvester
"""
import fatiando as ft
import cProfile

print ft.log.header()
model = [ft.msh.ddd.Prism(250, 750, 250, 750, 200, 700, {'density':1000})]
shape = (25, 25)
bounds = [0, 1000, 0, 1000, 0, 1000]
area = bounds[0:4]
xp, yp, zp = ft.grd.regular(area, shape, z=-1)
gz = ft.pot.prism.gz(xp, yp, zp, model)
mesh = ft.msh.ddd.PrismMesh(bounds, (25, 25, 25))
dms = ft.pot.harvester.wrapdata(mesh, xp, yp, zp, gz=gz)
seeds = ft.pot.harvester.sow_prisms([[500, 500, 450]], {'density':[1000]},
    mesh, mu=0.1, delta=0.00001)
cProfile.run('potential.harvester.harvest(dms, seeds)')
