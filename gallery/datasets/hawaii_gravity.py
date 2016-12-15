"""
Hawaii gravity data
-------------------

The :mod:`fatiando.datasets` package includes some data sets to make it easier
to try things out in Fatiando.

This example shows the gravity data from Hawaii.

"""
from __future__ import print_function
from fatiando.datasets import fetch_hawaii_gravity
import numpy as np
import matplotlib.pyplot as plt

# Load the gravity data from Hawaii
data = fetch_hawaii_gravity()

# The data are packaged in a dictionary. Look at the keys to see what is
# available.
print('Data keys:', data.keys())

# There are some metadata included
print('\nMetadata:\n')
print(data['metadata'])

# Let's plot all of it using the UTM x and y coordinates
shape = data['shape']
X, Y = data['x'].reshape(shape)/1000, data['y'].reshape(shape)/1000

fig = plt.figure(figsize=(14, 8))
plt.rcParams['font.size'] = 10

ax = plt.subplot(2, 3, 2)
ax.set_title('Raw gravity of Hawaii')
tmp = ax.contourf(Y, X, data['gravity'].reshape(shape), 60,
                  cmap='Reds')
fig.colorbar(tmp, ax=ax, pad=0, aspect=30).set_label('mGal')

ax = plt.subplot(2, 3, 3)
ax.set_title('Topography')
scale = np.abs([data['topography'].min(), data['topography'].max()]).max()
tmp = ax.contourf(Y, X, data['topography'].reshape(shape), 60,
                  cmap='terrain', vmin=-scale, vmax=scale)
fig.colorbar(tmp, ax=ax, pad=0, aspect=30).set_label('m')

ax = plt.subplot(2, 3, 4)
ax.set_title('Gravity disturbance')
scale = np.abs([data['disturbance'].min(), data['disturbance'].max()]).max()
tmp = ax.contourf(Y, X, data['disturbance'].reshape(shape), 60,
                  cmap='RdBu_r', vmin=-scale, vmax=scale)
fig.colorbar(tmp, ax=ax, pad=0, aspect=30).set_label('mGal')

# The disturbance without the effects of topography (calculated using the
# Bouguer plate)
ax = plt.subplot(2, 3, 5)
ax.set_title('Topography-free disturbance (Bouguer)')
tmp = ax.contourf(Y, X, data['topo-free-bouguer'].reshape(shape), 60,
                  cmap='viridis')
fig.colorbar(tmp, ax=ax, pad=0, aspect=30).set_label('mGal')

# The disturbance without the effects of topography (calculated using a
# tesseroid model of the topography)
ax = plt.subplot(2, 3, 6)
ax.set_title('Topography-free disturbance (full)')
tmp = ax.contourf(Y, X, data['topo-free'].reshape(shape), 60,
                  cmap='viridis')
fig.colorbar(tmp, ax=ax, pad=0, aspect=30).set_label('mGal')

plt.tight_layout()
plt.show()
