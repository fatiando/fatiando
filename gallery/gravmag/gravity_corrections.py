"""
Gravity corrections (normal gravity and Bouguer)
------------------------------------------------------

The :mod:`fatiando.gravmag` has function for performing gravity corrections to
obtain the gravity disturbance (measured gravity minus normal gravity).
Particularly, there are functions for calculating normal gravity at any height
using a closed-form formula instead of using the free-air approximation.

This example calculates the gravity disturbance and the topography-free
disturbance (what is usually called the "Bouguer anomaly" in geophysics) using
raw gravity data from Hawaii.

"""
from fatiando.datasets import fetch_hawaii_gravity
from fatiando.gravmag import normal_gravity
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Load and unpack gravity data from Hawaii
data = fetch_hawaii_gravity()
s, n, w, e = data['area']
shape = data['shape']
lat, lon, height = data['lat'], data['lon'], data['height']
g, topo = data['gravity'], data['topography']

# Use the closed form of the normal gravity to calculate
# it at the observation height. This is better than using
# the free-air approximation.
disturbance = g - normal_gravity.gamma_closed_form(lat, height)
# Use a Bouguer plate to remove the effect of topography
bouguer = disturbance - normal_gravity.bouguer_plate(topo)

# Plot the data using a Mercator projection
bm = Basemap(projection='merc',
             llcrnrlat=s, llcrnrlon=w,
             urcrnrlat=n, urcrnrlon=e)
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
ax = axes[0, 0]
ax.set_title('Raw gravity of Hawaii')
tmp = bm.contourf(lon, lat, g, 60, cmap='Reds',
                  latlon=True, tri=True, ax=ax)
fig.colorbar(tmp, ax=ax, pad=0, aspect=30).set_label('mGal')
ax = axes[0, 1]
ax.set_title('Topography')
scale = np.abs([topo.min(), topo.max()]).max()
tmp = bm.contourf(lon, lat, topo, 60, cmap='terrain',
                  vmin=-scale, vmax=scale,
                  latlon=True, tri=True, ax=ax)
fig.colorbar(tmp, ax=ax, pad=0, aspect=30).set_label('m')
ax = axes[1, 0]
ax.set_title('Gravity disturbance')
scale = np.abs([disturbance.min(), disturbance.max()]).max()
tmp = bm.contourf(lon, lat, disturbance, 60, cmap='RdBu_r',
                  vmin=-scale, vmax=scale,
                  latlon=True, tri=True, ax=ax)
fig.colorbar(tmp, ax=ax, pad=0, aspect=30).set_label('mGal')
ax = axes[1, 1]
ax.set_title('Topography-free disturbance')
tmp = bm.contourf(lon, lat, bouguer, 60, cmap='viridis',
                  latlon=True, tri=True, ax=ax)
fig.colorbar(tmp, ax=ax, pad=0, aspect=30).set_label('mGal')
plt.tight_layout()
plt.show()
