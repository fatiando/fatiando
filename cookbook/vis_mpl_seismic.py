# This Python file uses the following encoding: utf-8
"""

Simple synthetic wedge model example using obspy utilities.

Convolve a sequence of spikes with a butterworth wavelet
showing the frequency tuning effect.

Save two segy files: wedge_spike.segy and wedge.segy

"""

import numpy as np
from fatiando.vis import mpl
from fatiando import utils
from scipy import signal

# create a synthetic wedge model 70 traces
rc = np.zeros((70, 90), dtype='float32')  # reflectivity series
rc[3:-3, 20] = 1.  # wedge top 80 ms
for i in xrange(3, 67, 1):  # wedge base 2ms:trace
    rc[i, 20 + int(i*0.45)] = -1.
traces = utils.matrix2stream(rc, header={'delta': 0.004})  # sample rate 4 ms
# save as a segy file
traces.write('wedge_spike.segy', format='SEGY')
del traces # delete the traces object
# read the file now
wedge = utils.read('wedge_spike.segy')
mpl.figure(figsize=(11, 7))
mpl.subplot(121)
mpl.seismic_wiggle(wedge, normalize=True)
mpl.title("Wedge spike model (wedge_spike.segy)", fontsize=11)
# filtering using a IIR (infinite impulse response) recursive filter
# band defined as f1=0.1Fn f2=0.4Fn [ 12.5 Hz, 50.0 Hz ]
b, a = signal.butter(8, (0.1, 0.4), btype='bandpass')
for trac in wedge:  # same as convolving traces with a wavelet (butterworth)
    trac.data = signal.filtfilt(b, a, trac.data)
    trac.data = trac.data.astype('float32')  # convert back to float32 to save
mpl.subplot(122)
mpl.seismic_image(wedge, cmap=mpl.pyplot.cm.jet, aspect='auto')
mpl.seismic_wiggle(wedge, scale=2.0)
mpl.title("Wedge convolved model (wedge.segy)", fontsize=11)
mpl.show()
# save as the final segy
wedge.write('wedge.segy', format='SEGY')


