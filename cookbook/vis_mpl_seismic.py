"""
Seismic: Simple synthetic wedge model creation and display.

Convolve a sequence of spikes with a ricker wavelet
showing the frequency tuning effect in a pinchout
"""

import urllib
import numpy as np
from scipy import signal, fftpack
from fatiando import utils, io
from fatiando.vis import mpl

dt = 0.002
ds = 10.
shp = (70, 560)
nz, nx = shp
vwedge = io.fromimage('pinchout.bmp', ranges=[2500., 3500.], shape=shp)  # m/s
# calculate times, space increment is 1
timesi = np.cumsum(ds/vwedge, axis=0)*2  # irregular sampled twt time
times = np.arange(0, np.max(timesi)+.3, dt)  # twt re-sampled to 2 ms
svwedge = np.zeros((len(times), nx))
for i in xrange(nx):  # time re-sample velocity model
    svwedge[:, i] = np.interp(times, timesi[:, i], vwedge[:, i])
zimp = np.ones(svwedge.shape)*1000.*svwedge  # rho = kg/m3, convert to impedance z = rho*v
# filter impedance? and downsample?
# calculate reflection coefficients
rc = (zimp[1:]-zimp[:-1])/(zimp[1:]+zimp[:-1])
# filtering using a ricker
mpl.figure(figsize=(11, 7))
mpl.subplot(221)
fir_wavelet = signal.ricker(255, 10.)
mpl.plot(fir_wavelet)
mpl.xlim(xmin=0, xmax=255)
mpl.subplot(222)
freqs = fftpack.fftfreq(255, dt)
fir_waveletw = fftpack.fft(fir_wavelet)
mpl.xlim(xmin=0, xmax=125)
mpl.plot(freqs, np.abs(fir_waveletw))
# same as convolving traces with a wavelet
rc = signal.lfilter(fir_wavelet, 1, rc, axis=0)
# create a synthetic wedge model 120 traces
rc = rc[:, ::2]
mpl.subplot(223)
traces = utils.matrix2stream(rc.transpose(), header={'delta': dt})  # sample rate 2.5 us
mpl.seismic_wiggle(traces, normalize=True)
mpl.seismic_image(traces, cmap=mpl.pyplot.cm.jet)
mpl.title("Wedge spike model", fontsize=11)
mpl.subplot(224)
mpl.imshow(vwedge, extent=[0, nx, 0, nz*ds], aspect=0.4)
#mpl.seismic_wiggle(traces, scale=2.0)
mpl.title("Wedge convolved model", fontsize=11)
mpl.show()
