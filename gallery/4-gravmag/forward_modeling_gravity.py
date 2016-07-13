"""
Gravity 3D forward modeling
---------------------------


"""
import numpy as np
import matplotlib.pyplot as plt
from fatiando import mesher, gridder
from fatiando.gravmag.forward import prism


model = mesher.Prism(-300, 300, -300, 300, -300, 300, {'density': 1000})
area = (-600, 600, -600, 600)
# x, y, z = gridder.regular(area, (101, 101), z=-150)
shape = (101, 101)
x, y, z = gridder.regular(area, shape , z=310)
fields = [
          prism.potential(x, y, z, model),
          prism.gx(x, y, z, model),
          prism.gy(x, y, z, model),
          prism.gz(x, y, z, model),
          prism.gxx(x, y, z, model),
          prism.gxy(x, y, z, model),
          prism.gxz(x, y, z, model),
          prism.gyy(x, y, z, model),
          prism.gyz(x, y, z, model),
          prism.gzz(x, y, z, model)
          ]
titles = [
    'potential', 'gx', 'gy', 'gz',
           'gxx',
          'gxy',
          'gxz', 'gyy', 'gyz', 'gzz'
          ]

print np.abs(fields[5]).max()

plt.figure(figsize=(8, 9))
plt.subplots_adjust(left=0.03, right=0.95, bottom=0.05, top=0.92, hspace=0.3)
plt.suptitle("Potential fields produced by a 3 prism model")
for i, field in enumerate(fields):
    plt.subplot(4, 3, i + 3)
    plt.axis('scaled')
    plt.title(titles[i])
    plt.pcolormesh(y.reshape(shape), x.reshape(shape), field.reshape(shape),
                   cmap='cubehelix')
    cb = plt.colorbar()
    plt.xlim(area[2:])
    plt.ylim(area[:2])
plt.show()
