"""
Potential: 3D forward modeling of total-field magnetic anomaly using rectangular
prisms (model with induced and remanent magnetization)
"""
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

bounds = [-5000, 5000, -5000, 5000, 0, 5000]
prisms = [
    ft.msh.ddd.Prism(-4000,-3000,-4000,-3000,0,2000,
        {'magnetization':2}),
    ft.msh.ddd.Prism(-1000,1000,-1000,1000,0,2000,
        {'magnetization':1}),
    # This prism has remanent magnetization because it's physical property
    # dict has inclination and declination
    ft.msh.ddd.Prism(2000,4000,3000,4000,0,2000,
        {'magnetization':3, 'inclination':-10, 'declination':45})]
# Create a regular grid at 100m height
shape = (200, 200)
area = bounds[:4]
xp, yp, zp = ft.grd.regular(area, shape, z=-500)
# Calculate the anomaly for a given regional field
tf = ft.pot.prism.tf(xp, yp, zp, prisms, 30, -15)
# Plot
ft.vis.figure()
ft.vis.title("Total-field anomaly (nT)")
ft.vis.axis('scaled')
ft.vis.contourf(yp, xp, tf, shape, 15)
ft.vis.colorbar()
ft.vis.xlabel('East y (km)')
ft.vis.ylabel('North x (km)')
ft.vis.m2km()
ft.vis.show()
# Show the prisms
ft.vis.figure3d()
ft.vis.prisms(prisms, 'magnetization')
ft.vis.axes3d(ft.vis.outline3d(bounds), ranges=[i*0.001 for i in bounds])
ft.vis.wall_north(bounds)
ft.vis.wall_bottom(bounds)
ft.vis.show3d()
