"""
Seismic: seismic plotting utilities of Obspy
"""
from fatiando.vis import mpl
from fatiando.utils import readSEGY
import urllib

# fetch sample SEGY data, near-offset marmousi data
url = 'http://dl.dropboxusercontent.com/s/i287ci4ww3w7gdt/marmousi_nearoffset.segy'
urllib.urlretrieve(url, 'marmousi_nearoffset.sgy')
# plot using wiggle
segyfile = readSEGY('marmousi_nearoffset.sgy')
mpl.seismic_wiggle(segyfile, scale=10**-4)
mpl.show()
# plot using image
mpl.seismic_image(segyfile)
mpl.show()