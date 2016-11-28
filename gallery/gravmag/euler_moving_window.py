"""
.. _gallery_euler_mw:

Euler deconvolution with a moving window
----------------------------------------

Euler deconvolution attempts to estimate the coordinates of simple (idealized)
sources from the input potential field data. There is a strong assumption that
the sources have simple geometries, like spheres, vertical pipes, vertical
planes, etc. So it wouldn't be much of a surprise if the solutions aren't great
when sources are complex.

Let's test the Euler deconvolution using a moving window scheme, a very common
approach used in all industry software. This is implemented in
:class:`fatiando.gravmag.euler.EulerDeconvMW`.


"""
from __future__ import print_function
from fatiando.gravmag import sphere, transform, euler
from fatiando import gridder, utils, mesher
import matplotlib.pyplot as plt

# Make some synthetic magnetic data to test our Euler deconvolution.
# The regional field
inc, dec = -45, 0
# Make a model of two spheres magnetized by induction only
model = [
    mesher.Sphere(x=-1000, y=-1000, z=1500, radius=1000,
                  props={'magnetization': utils.ang2vec(2, inc, dec)}),
    mesher.Sphere(x=1000, y=1500, z=1000, radius=1000,
                  props={'magnetization': utils.ang2vec(1, inc, dec)})]

print("Centers of the model spheres:")
print(model[0].center)
print(model[1].center)

# Generate some magnetic data from the model
shape = (100, 100)
area = [-5000, 5000, -5000, 5000]
x, y, z = gridder.regular(area, shape, z=-150)
data = sphere.tf(x, y, z, model, inc, dec)

# We also need the derivatives of our data
xderiv = transform.derivx(x, y, data, shape)
yderiv = transform.derivy(x, y, data, shape)
zderiv = transform.derivz(x, y, data, shape)

# Now we can run our Euler deconv solver on a moving window over the data.
# Each window will produce an estimated point for the source.
# We use a structural index of 3 to indicate that we think the sources are
# spheres.

# Run the Euler deconvolution on moving windows to produce a set of solutions
# by running the solver on 10 x 10 windows of size 1000 x 1000 m
solver = euler.EulerDeconvMW(x, y, z, data, xderiv, yderiv, zderiv,
                             structural_index=3, windows=(10, 10),
                             size=(1000, 1000))
# Use the fit() method to obtain the estimates
solver.fit()

# The estimated positions are stored as a list of [x, y, z] coordinates
# (actually a 2D numpy array)
print('Kept Euler solutions after the moving window scheme:')
print(solver.estimate_)

# Plot the solutions on top of the magnetic data. Remember that the true depths
# of the center of these sources is 1500 m and 1000 m.

plt.figure(figsize=(6, 5))
plt.title('Euler deconvolution with a moving window')
plt.contourf(y.reshape(shape), x.reshape(shape), data.reshape(shape), 30,
             cmap="RdBu_r")
plt.scatter(solver.estimate_[:, 1], solver.estimate_[:, 0],
            s=50, c=solver.estimate_[:, 2], cmap='cubehelix')
plt.colorbar(pad=0).set_label('Depth (m)')
plt.xlim(area[2:])
plt.ylim(area[:2])
plt.tight_layout()
plt.show()
