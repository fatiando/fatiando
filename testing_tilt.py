from __future__ import division
import numpy as np
from numpy.testing import assert_allclose
from fatiando.gravmag import transform, prism
from fatiando import gridder, utils
from fatiando.mesher import Prism
from fatiando.vis import mpl
import matplotlib.pyplot as plt


def find_nearest(array, value):
    "Find the nearest value in the that matches `value`."
    idx = (np.abs(array-value)).argmin()
    return array[idx]


model = [Prism(-1000, 1000, -500, 500, 0, 2000, {'magnetization': 3200})]
shape = (300, 300)
x, y, z = gridder.regular([-5000, 5000, -5000, 5000], shape, z=-100)
inc, dec = -90, 0
potential = prism.tf(x, y, z, model, inc, dec)
tilt_angle = transform.tilt(x, y, potential, shape)


inside_x = [find_nearest(y, np.random.uniform(-900, 900)) for i in range(100)]
inside_y = [find_nearest(x, np.random.uniform(-450, 450)) for i in range(100)]
outside_neg_x = [find_nearest(y, np.random.uniform(-3000, -1100))
                 for i in range(100)]
outside_neg_y = [find_nearest(x, np.random.uniform(-3000, -1000))
                 for i in range(100)]
outside_pos_x = [find_nearest(y, np.random.uniform(1100, 3000))
                 for i in range(100)]
outside_pos_y = [find_nearest(x, np.random.uniform(600, 3000))
                 for i in range(100)]


x_indices = []
y_indices = []
indices = []
for i in range(len(inside_x)):
    x_indices.append(np.where(x == inside_x[i])[0][0])

for i in range(len(inside_y)):
    y_indices.append(np.where(y == inside_y[i])[0][0])

for i in range(len(x_indices)):
    indices.append(x_indices[i] + y_indices[i])

assert np.any(tilt_angle[indices] > 0), \
    "Tilt values within model too low: {}".format(tilt_angle[indices])

x_values_in = np.array(x[x_indices])
y_values_in = np.array(y[y_indices])

x_indices = []
y_indices = []
indices = []
for i in range(len(outside_neg_x)):
    x_indices.append(np.where(x == outside_neg_x[i])[0][0])
    x_indices.append(np.where(x == outside_neg_x[i])[0][0])
    x_indices.append(np.where(x == outside_pos_x[len(outside_neg_x)-1-i])
                     [0][0])
    x_indices.append(np.where(x == outside_pos_x[len(outside_neg_x)-1-i])
                     [0][0])

for i in range(len(outside_neg_y)):
    y_indices.append(np.where(y == outside_pos_y[i])[0][0])
    y_indices.append(np.where(y == outside_neg_y[len(outside_neg_y)-1-i])
                     [0][0])
    y_indices.append(np.where(y == outside_pos_y[i])
                     [0][0])
    y_indices.append(np.where(y == outside_neg_y[len(outside_neg_y)-1-i])
                     [0][0])

for i in range(len(x_indices)):
    indices.append(x_indices[i] + y_indices[i])
    indices.append(x_indices[i] + y_indices[len(x_indices)-1-i])
    indices.append(x_indices[len(x_indices)-1-i] + y_indices[i])
assert np.any(tilt_angle[indices] < 0), \
    "Tilt values outside model too high: {}".format(tilt_angle[indices])

x_values = np.array(x[x_indices])
y_values = np.array(y[y_indices])
x_values_rev = x_values[::-1]
y_values_rev = y_values[::-1]

plt.figure()
plt.subplot(1, 2, 1)
plt.title("Original total field anomaly")
plt.axis('scaled')
plt.contourf(y.reshape(shape), x.reshape(shape), potential.reshape(shape), 30,
             cmap=plt.cm.RdBu_r)
plt.colorbar(orientation='horizontal').set_label('nT')
plt.subplot(1, 2, 2)
plt.title("Tilt")
plt.axis('scaled')
plt.contourf(y.reshape(shape), x.reshape(shape), tilt_angle.reshape(shape), 30,
             cmap=plt.cm.RdBu_r)
plt.colorbar(orientation='horizontal').set_label('Radian')
plt.scatter(y_values, x_values)
plt.scatter(y_values_rev, x_values)
plt.scatter(y_values, x_values_rev)
plt.scatter(y_values_in, x_values_in)
mpl.show()

"""
mpl.figure()
mpl.subplot(1, 2, 1)
mpl.title("Original total field anomaly")
mpl.axis('scaled')
mpl.contourf(y, x, potential, shape, 30, cmap=mpl.cm.RdBu_r)
mpl.colorbar(orientation='horizontal').set_label('nT')
mpl.m2km()
mpl.subplot(1, 2, 2)
mpl.title("Tilt")
mpl.axis('scaled')
mpl.contourf(y, x, tilt_angle, shape, 30, cmap=mpl.cm.RdBu_r)
mpl.colorbar(orientation='horizontal').set_label('Radian')
plt.scatter(y_indices, x_indices)
mpl.m2km()
mpl.show()
"""
