"""
Seismic: plotting a seismic section from a SEGY

Uses ObsPy package for SEGY file reading
You can find instructions for installing Obspy at http://obspy.org/

"""
from fatiando.vis import mpl
from obspy.segy import segy
import urllib
import numpy as np

# fetch sample SEGY data, near-offset marmousi data
url = "http://dl.dropboxusercontent.com/" \
      "s/i287ci4ww3w7gdt/marmousi_nearoffset.segy"
urllib.urlretrieve(url, 'marmousi_nearoffset.segy')
# We'll use the ObsPy library to load the SEGY data"
segyfile = segy.readSEGY('marmousi_nearoffset.segy')
# turn ObsPy Stream in a matrix of traces
# first dimension time, second dimension traces
ntraces = len(segyfile.traces)
nsamples = len(segyfile.traces[0].data)
mtraces = np.zeros((nsamples, ntraces))
i = 0
for tr in segyfile.traces:
    mtraces[:, i] = tr.data[:]
    i += 1
# make plots
mpl.figure()
mpl.subplot(2, 1, 1)
mpl.ylabel('time (seconds)')
mpl.title("Seismic wiggle plot", fontsize=13, family='sans-serif',
          weight='bold')
# plot using wiggle
mpl.seismic_wiggle(mtraces, scale=10**-4)
mpl.subplot(2, 1, 2)
mpl.ylabel('time (seconds)')
mpl.title("Seismic image plot", fontsize=13, family='sans-serif',
          weight='bold')
# plot using image
mpl.seismic_image(mtraces, aspect='auto')
mpl.show()
