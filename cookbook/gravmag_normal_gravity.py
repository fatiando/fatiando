"""
GravMag: Calculate the gravity disturbance and Bouguer anomaly for Hawaii
"""
from fatiando.gravmag import normal_gravity
from fatiando.vis import mpl
import numpy as np
import urllib

# Download the gravity and topography data
url = 'https://raw.githubusercontent.com/leouieda/geofisica1/master/data/'
urllib.urlretrieve(url + 'eigen-6c3stat-havai.gdf',
                   filename='eigen-6c3stat-havai.gdf')
urllib.urlretrieve(url + 'etopo1-havai.gdf',
                   filename='etopo1-havai.gdf')
# Load them with numpy
lon, lat, height, gravity = np.loadtxt('eigen-6c3stat-havai.gdf', skiprows=34,
                                       unpack=True)
topo = np.loadtxt('etopo1-havai.gdf', skiprows=30, usecols=[-1], unpack=True)
shape = (151, 151)
area = (lon.min(), lon.max(), lat.min(), lat.max())

# First, lets calculate the gravity disturbance (e.g., the free-air anomaly)
# We'll do this using the closed form of the normal gravity for the WGS84
# ellipsoid
gamma = normal_gravity.gamma_closed_form(lat, height)
disturbance = gravity - gamma

# Now we can remove the effect of the Bouguer plate to obtain the Bouguer
# anomaly. We'll use the standard densities of 2.67 g.cm^-3 for crust and 1.04
# g.cm^-3 for water.
bouguer = disturbance - normal_gravity.bouguer_plate(topo)

mpl.figure(figsize=(14, 3.5))
bm = mpl.basemap(area, projection='merc')
mpl.subplot(131)
mpl.title('Gravity (mGal)')
mpl.contourf(lon, lat, gravity, shape, 60, cmap=mpl.cm.Reds, basemap=bm)
mpl.colorbar(pad=0)
mpl.subplot(132)
mpl.title('Gravity disturbance (mGal)')
amp = np.abs(disturbance).max()
mpl.contourf(lon, lat, disturbance, shape, 60, cmap=mpl.cm.RdBu_r, basemap=bm,
             vmin=-amp, vmax=amp)
mpl.colorbar(pad=0)
mpl.subplot(133)
mpl.title('Bouguer anomaly (mGal)')
mpl.contourf(lon, lat, bouguer, shape, 60, cmap=mpl.cm.Reds, basemap=bm)
mpl.colorbar(pad=0)
mpl.show()
