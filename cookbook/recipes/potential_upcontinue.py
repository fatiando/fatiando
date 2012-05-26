"""
Upcontinue noisy gz data using the analytical formula
"""
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

log.info("Generating synthetic data")
prisms = [ft.msh.ddd.Prism(-3000,-2000,-3000,-2000,500,2000,{'density':1000}),
          ft.msh.ddd.Prism(-1000,1000,-1000,1000,0,2000,{'density':-800}),
          ft.msh.ddd.Prism(1000,3000,2000,3000,0,1000,{'density':500})]
area = (-5000, 5000, -5000, 5000)
shape = (25, 25)
z0 = -100
xp, yp, zp = ft.grd.regular(area, shape, z=z0)
gz = ft.utils.contaminate(ft.pot.prism.gz(xp, yp, zp, prisms), 0.5)

# Now do the upward continuation using the analytical formula
height = 2000
dims = ft.grd.spacing(area, shape)
gzcont = ft.pot.trans.upcontinue(gz, z0, height, xp, yp, dims)

log.info("Computing true values at new height")
gztrue = ft.pot.prism.gz(xp, yp, zp - height, prisms)

log.info("Plotting")
ft.vis.figure(figsize=(14,6))
ft.vis.subplot(1, 2, 1)
ft.vis.title("Original")
ft.vis.axis('scaled')
ft.vis.contourf(xp, yp, gz, shape, 15)
ft.vis.contour(xp, yp, gz, shape, 15)
ft.vis.subplot(1, 2, 2)
ft.vis.title("Continued + true")
ft.vis.axis('scaled')
levels = ft.vis.contour(xp, yp, gzcont, shape, 12, color='b',
    label='Continued', style='dashed')
ft.vis.contour(xp, yp, gztrue, shape, levels, color='r', label='True',
    style='solid')
ft.vis.legend()
ft.vis.show()
