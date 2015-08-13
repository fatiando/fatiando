"""
GravMag: Calculating the radial average of a Bouguer Anomaly Power Spectrum
"""

import numpy as np
import matplotlib.pyplot as plt
from fatiando.datasets import fetch_bouguer_alps_egm
from fatiando.gridder import load_surfer
from fatiando.gravmag import transform

# Load Bouguer Anomaly Data
fetch_bouguer_alps_egm(fname="alps-bouguer.grd")
x, y, bg, shape = load_surfer("alps-bouguer.grd")

# Puts Data in 2d array grid
y = y.reshape(shape)  # Easting
x = x.reshape(shape)  # Northing
bg = bg.reshape(shape)

# Cut the data to a square grid
x, y, bg = x[0:181, 19:200], y[0:181, 19:200], bg[0:181, 19:200]
dx = abs(x[1, 0] - x[0, 0])

# Calculates the Power Spectrum and frequencies
kx = 2*np.pi*np.fft.fftfreq(181, dx)
ky, kx = np.meshgrid(kx, kx)
fft_bg = np.fft.fft2(bg)
log_power_spectrum = np.log(abs(fft_bg)**2)

# Calculates the Radial Average
k, average, rms = transform.radial_average(log_power_spectrum, dx)

# PLOTS
plt.axes(aspect='equal')
plt.contourf(y, x, bg, 150)
plt.colorbar(label='mGal')
plt.title("Bouguer Anomaly")
plt.xlabel("m")
plt.ylabel("m")
plt.show()

kx = np.fft.fftshift(kx)
ky = np.fft.fftshift(ky)
log_power_spectrum = np.fft.fftshift(log_power_spectrum)
plt.axes(aspect='equal')
plt.contourf(ky, kx, log_power_spectrum, 150)
plt.colorbar()
plt.title("Log Power Spectrum")
plt.xlabel("rad/km")
plt.ylabel("rad/km")
plt.show()

plt.errorbar(k, average, yerr=rms, fmt='.')
plt.xlabel("k [rad/km]")
plt.ylabel("Rad Averaged Log Power Spectrum")
plt.title("Radially Averaged Log Power Spectrum")
plt.show()
