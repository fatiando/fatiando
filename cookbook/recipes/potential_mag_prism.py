"""
Calculate the total-field anomaly of a prism model with induced magnetization
only.
"""
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

prisms = [
    ft.msh.ddd.Prism(-4000,-3000,-4000,-3000,0,2000,
        {'magnetization':20}),
    ft.msh.ddd.Prism(-1000,1000,-1000,1000,0,2000,
        {'magnetization':10}),
    ft.msh.ddd.Prism(2000,4000,3000,4000,0,2000,
        {'magnetization':30, 'inclination':-10, 'declination':45})]
# Create a regular grid at 100m height
shape = (200, 200)
area = (-5000, 5000, -5000, 5000)
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
