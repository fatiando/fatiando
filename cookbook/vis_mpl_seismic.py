"""
Seismic: seismic plotting utilities using Obspy for SEGY reading
"""
from fatiando.vis import mpl
from obspy.segy import segy
import urllib
import numpy as np

# fetch sample SEGY data, near-offset marmousi data
url = "http://dl.dropboxusercontent.com/" \
      "s/i287ci4ww3w7gdt/marmousi_nearoffset.segy"
urllib.urlretrieve(url, 'marmousi_nearoffset.segy')
segyfile = segy.readSEGY('marmousi_nearoffset.segy')
# turn Obspy Stream in a matrix of traces
# first dimension time second dimension traces
ntraces = len(segyfile.traces)
nsamples = len(segyfile.traces[0].data)
mtraces = np.zeros((nsamples, ntraces))
i = 0
for tr in segyfile.traces:
    mtraces[:, i] = tr.data[:]
    i += 1
# plot using wiggle
mpl.seismic_wiggle(mtraces, scale=10**-4)
mpl.show()
# plot using image
mpl.seismic_image(mtraces)
mpl.show()