"""
Calculate the total-field anomaly of a prism model with induced magnetization
only.
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
ft.vis.contourf(yp*0.001, xp*0.001, tf, shape, 15)
ft.vis.colorbar()
ft.vis.xlabel('East y (km)')
ft.vis.ylabel('North x (km)')
ft.vis.show()
# Show the prisms
ft.vis.figure3d()
ft.vis.prisms(prisms, 'magnetization')
ft.vis.axes3d(ft.vis.outline3d(bounds), ranges=[i*0.001 for i in bounds])
ft.vis.wall_north(bounds)
ft.vis.wall_bottom(bounds)
ft.vis.show3d()
