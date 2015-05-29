"""
GravMag: Reduction to the pole of a total field anomaly using FFT
"""
from fatiando import mesher, gridder, utils
from fatiando.gravmag import prism, transform
from fatiando.vis import mpl

# Direction of the Geomagnetic field
inc, dec = -60, 0
# Make a model with only induced magnetization
model = [mesher.Prism(-100, 100, -100, 100, 0, 2000,
                      {'magnetization': utils.ang2vec(10, inc, dec)})]
area = (-5000, 5000, -5000, 5000)
shape = (100, 100)
z0 = -500
x, y, z = gridder.regular(area, shape, z=z0)
tf = utils.contaminate(prism.tf(x, y, z, model, inc, dec),
                       1, seed=0)
# Reduce to the pole using FFT. Since there is only induced magnetization, the
# magnetization direction (sinc and sdec) is the same as the geomagnetic field
pole = transform.reduce_to_pole(x, y, tf, shape, inc, dec, sinc=inc, sdec=dec)
# Calculate the true value at the pole for comparison
true = prism.tf(x, y, z, model, 90, 0, pmag=utils.ang2vec(10, 90, 0))

fig, axes = mpl.subplots(1, 3, figsize=(14, 4))
for ax in axes:
    ax.set_aspect('equal')
mpl.sca(axes[0])
mpl.title("Original total field anomaly")
mpl.contourf(y, x, tf, shape, 30, cmap=mpl.cm.RdBu_r)
mpl.colorbar(pad=0).set_label('nT')
mpl.m2km()
mpl.sca(axes[1])
mpl.title("True value at pole")
mpl.contourf(y, x, true, shape, 30, cmap=mpl.cm.RdBu_r)
mpl.colorbar(pad=0).set_label('nT')
mpl.m2km()
mpl.sca(axes[2])
mpl.title("Reduced to the pole")
mpl.contourf(y, x, pole, shape, 30, cmap=mpl.cm.RdBu_r)
mpl.colorbar(pad=0).set_label('nT')
mpl.m2km()
mpl.tight_layout()
mpl.show()
