"""
GravMag: Forward modeling of the gravity anomaly and gravity gradient tensor
using model
"""
from fatiando import mesher, gridder
from fatiando.gravmag import sphere
from fatiando.vis import mpl

model = [mesher.Sphere(0, 0, 2000, 1000, {'density': 1000})]
area = (-5000, 5000, -5000, 5000)
shape = (100, 100)
x, y, z = gridder.regular(area, shape, z=-100)
gz = sphere.gz(x, y, z, model)
tensor = [sphere.gxx(x, y, z, model),
          sphere.gxy(x, y, z, model),
          sphere.gxz(x, y, z, model),
          sphere.gyy(x, y, z, model),
          sphere.gyz(x, y, z, model),
          sphere.gzz(x, y, z, model)]
mpl.figure()
mpl.axis('scaled')
mpl.title('gz')
mpl.contourf(y, x, gz, shape, 15)
mpl.colorbar()
mpl.m2km()
mpl.figure()
titles = ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
for i, field in enumerate(tensor):
    mpl.subplot(2, 3, i + 1)
    mpl.axis('scaled')
    mpl.title(titles[i])
    levels = mpl.contourf(y, x, field, shape, 15)
    mpl.colorbar()
    mpl.m2km()
mpl.show()
