"""
GravMag: Upward continuation of noisy gz data
"""
from fatiando import mesher, gridder, utils
from fatiando.gravmag import prism, transform, fourier
from fatiando.vis import mpl

model = [
    mesher.Prism(-3000, -2000, -3000, -2000, 500, 2000, {'density': 1000}),
    mesher.Prism(-1000, 1000, -1000, 1000, 0, 2000, {'density': -800}),
    mesher.Prism(1000, 3000, 2000, 3000, 0, 1000, {'density': 500})]
area = (-5000, 5000, -5000, 5000)
shape = (50, 50)
z0 = -100
xp, yp, zp = gridder.regular(area, shape, z=z0)
gz = utils.contaminate(prism.gz(xp, yp, zp, model), 0.5, seed=0)

# The how much higher to go
height = 2000

# Now do the upward continuation using the analytical formula
dims = gridder.spacing(area, shape)
gzcont = transform.upcontinue(gz, height, xp, yp, dims)

# and using the Fourier transform
gzcontf = fourier.upcontinue(xp, yp, gz, shape, height)

gztrue = prism.gz(xp, yp, zp - height, model)

mpl.figure(figsize=(14, 4))
mpl.subplot(1, 3, 1)
mpl.title("Original")
mpl.axis('scaled')
mpl.contourf(xp, yp, gz, shape, 15)
mpl.contour(xp, yp, gz, shape, 15)
mpl.subplot(1, 3, 2)
mpl.title("Continued (Analytical)")
mpl.axis('scaled')
levels = mpl.contour(xp, yp, gzcont, shape, 12, color='b',
                     label='Continued', style='dashed')
mpl.subplot(1, 3, 3)
mpl.title("Continued (Fourier)")
mpl.axis('scaled')
levels = mpl.contour(xp, yp, gzcontf, shape, 12, color='b',
                     label='Continued', style='dashed')
mpl.show()
