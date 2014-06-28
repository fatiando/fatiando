"""
GravMag: Generate noise-corrupted gravity gradient tensor data
"""
from fatiando import mesher, gridder, utils
from fatiando.gravmag import prism
from fatiando.vis import mpl

model = [mesher.Prism(-1000, 1000, -1000, 1000, 0, 2000, {'density': 1000})]
shape = (100, 100)
xp, yp, zp = gridder.regular((-5000, 5000, -5000, 5000), shape, z=-200)
components = [prism.gxx, prism.gxy, prism.gxz,
              prism.gyy, prism.gyz, prism.gzz]
print "Calculate the tensor components and contaminate with 5 Eotvos noise"
ftg = [utils.contaminate(comp(xp, yp, zp, model), 5.0) for comp in components]

print "Plotting..."
mpl.figure(figsize=(14, 6))
mpl.suptitle("Contaminated FTG data")
names = ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
for i, data in enumerate(ftg):
    mpl.subplot(2, 3, i + 1)
    mpl.title(names[i])
    mpl.axis('scaled')
    levels = mpl.contourf(xp * 0.001, yp * 0.001, data, (100, 100), 12)
    mpl.colorbar()
    mpl.contour(xp * 0.001, yp * 0.001, data, shape, levels, clabel=False)
mpl.show()
