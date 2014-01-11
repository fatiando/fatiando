# This Python file uses the following encoding: utf-8
"""

Simple synthetic wedge model (Frequency tuning example)
example using obspy utilities.

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
mpl.seismic_wiggle(traces, normalize=True)
mpl.show()
# save as a segy file
traces.write('wedge.segy', format='SEGY')
del traces # delete the traces object
# read the file now
wedge = utils.read('wedge.segy')
# filtering using a IIR (infinite impulse response) recursive filter
# band defined as f1=0.1Fn f2=0.4Fn [ 12.5 Hz, 50.0 Hz ]
b, a = signal.butter(8, (0.1, 0.4), btype='bandpass')
for trac in wedge:  # convolve with the traces
    trac.data = signal.filtfilt(b, a, trac.data)
    trac.data = trac.data.astype('float32')  # convert back to float32 to save
mpl.seismic_image(wedge, cmap=mpl.pyplot.cm.jet)
mpl.seismic_wiggle(wedge, scale=2.0)
mpl.show()
# save as the final segy
wedge.write('wedge.segy', format='SEGY')


