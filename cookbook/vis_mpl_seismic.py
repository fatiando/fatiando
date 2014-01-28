"""
Seismic: Simple synthetic wedge model creation and display.

Convolve a sequence of spikes with a butterworth wavelet
showing the frequency tuning effect in a pinchout
"""

import urllib
import numpy as np
from scipy import signal, fftpack
from fatiando import utils, io
from fatiando.vis import mpl

# ... load figure
urllib.urlretrieve(
    'https://lh6.googleusercontent.com/-ehOIzWKLHo4/UugZOJ71gFI/AAAAAAAADrg/dSTP2OozSbw/w300-h150-no/pinchout.bmp',
    'wedge.bmp')
vwedge = io.fromimage('wedge.png', ranges=[2500., 3500.], shape=(300, 150))  # m/s
zimp = np.ones((300, 150))*1000*vwedge  # rho = kg/m3, convert to impedance z = rho*v
# and reflection coefficients

# create a synthetic wedge model 70 traces
rc = np.zeros((90, 90), dtype='float32')  # reflectivity series
rc[3:-3, 20] = 0.5  # wedge top 80 ms
rc[3:13, 20] = -1  # wedge top 80 ms
for i in xrange(13, 87, 1):  # wedge base 2ms:trace
    rc[i, 20 + int((i-10)*0.45)] = -1
traces = utils.matrix2stream(rc, header={'delta': 0.004})  # sample rate 4 ms
# filtering using a IIR (infinite impulse response) recursive filter
# band defined as f1=0.1Fn f2=0.4Fn [ 12.5 Hz, 50.0 Hz ]
b, a = signal.butter(12, (0.1, 0.4), btype='bandpass')
mpl.figure(figsize=(11, 7))
mpl.subplot(221)
spike = np.zeros(101)
spike[50] = 1.
butter_wavelet = signal.filtfilt(b, a, spike)
mpl.plot(butter_wavelet)
mpl.subplot(222)
freqs = fftpack.fftfreq(101, 0.004)
butter_waveletW = fftpack.fft(butter_wavelet)
mpl.plot(freqs,np.abs(butter_waveletW))
mpl.xlim(xmin=0,xmax=125.)
mpl.subplot(223)
mpl.seismic_wiggle(traces, normalize=True)
mpl.title("Wedge spike model", fontsize=11)
for trac in traces:  # same as convolving traces with a wavelet (butterworth)
    trac.data = signal.filtfilt(b, a, trac.data)
mpl.subplot(224)
mpl.seismic_image(traces, cmap=mpl.pyplot.cm.jet, aspect='auto')
mpl.seismic_wiggle(traces, scale=2.0)
mpl.title("Wedge convolved model", fontsize=11)
mpl.show()

