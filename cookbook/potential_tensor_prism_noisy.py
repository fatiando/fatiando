"""
Potential: Generate noise-corrupted gravity gradient tensor data
"""
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

prisms = [ft.msh.ddd.Prism(-1000,1000,-1000,1000,0,2000,{'density':1000})]
shape = (100,100)
xp, yp, zp = ft.grd.regular((-5000, 5000, -5000, 5000), shape, z=-200)
components = [ft.pot.prism.gxx, ft.pot.prism.gxy, ft.pot.prism.gxz,
              ft.pot.prism.gyy, ft.pot.prism.gyz, ft.pot.prism.gzz]
log.info("Calculate the tensor components and contaminate with 5 Eotvos noise")
ftg = [ft.utils.contaminate(comp(xp, yp, zp, prisms), 5.0) for comp in components]

log.info("Plotting...")
ft.vis.figure(figsize=(14,6))
ft.vis.suptitle("Contaminated FTG data")
names = ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
for i, data in enumerate(ftg):
    ft.vis.subplot(2,3,i+1)
    ft.vis.title(names[i])
    ft.vis.axis('scaled')
    levels = ft.vis.contourf(xp*0.001, yp*0.001, data, (100,100), 12)
    ft.vis.colorbar()
    ft.vis.contour(xp*0.001, yp*0.001, data, shape, levels, clabel=False)
ft.vis.show()
