"""
GravMag: Upward continuation of noisy gz data
"""
from fatiando import mesher, gridder, utils
from fatiando.gravmag import prism, transform
from fatiando.vis import mpl
import numpy as np

model = [
    mesher.Prism(-3000, -2000, -3000, -2000, 500, 2000, {'density': 1000}),
    mesher.Prism(-1000, 1000, -1000, 1000, 0, 2000, {'density': -800}),
    mesher.Prism(1000, 3000, 2000, 3000, 0, 1000, {'density': 900})]
area = (-5000, 5000, -5000, 5000)
shape = (50, 50)
z0 = -100
x, y, z = gridder.regular(area, shape, z=z0)
gz = utils.contaminate(prism.gz(x, y, z, model), 0.5, seed=0)

height = 1000  # How much higher to go
gzcontf = transform.upcontinue(x, y, gz, shape, height)

# Compute the true value at the new height for comparison
gztrue = prism.gz(x, y, z - height, model)

args = dict(shape=shape, levels=20, cmap=mpl.cm.RdBu_r)
fig, axes = mpl.subplots(1, 3, figsize=(12, 3.5))
axes = axes.ravel()
mpl.sca(axes[0])
mpl.title("Original")
mpl.axis('scaled')
mpl.contourf(x, y, gz, **args)
mpl.colorbar(pad=0).set_label('mGal')
mpl.m2km()
mpl.sca(axes[1])
mpl.title('True higher')
mpl.axis('scaled')
mpl.contourf(y, x, gztrue, **args)
mpl.colorbar(pad=0).set_label('mGal')
mpl.m2km()
mpl.sca(axes[2])
mpl.title("Continued (Fourier)")
mpl.axis('scaled')
mpl.contourf(y, x, gzcontf, **args)
mpl.colorbar(pad=0).set_label('mGal')
mpl.m2km()
mpl.tight_layout()
mpl.show()
