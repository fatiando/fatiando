"""
Datasets: Fetch the CRUST2.0 model, convert it to tesseroids and calculate its
gravity signal in parallel
"""
import time
from multiprocessing import Pool
from fatiando import datasets
from fatiando.mesher import Tesseroid
from fatiando.vis import mpl, myv

# Get the data from their website and convert it to tesseroids
# Will download the archive and save it with the default name
archive = datasets.fetch_crust2()
model = datasets.crust2_to_tesseroids(archive)

# Plot the tesseroid model
myv.figure(zdown=False)
myv.tesseroids(model, 'density')
myv.continents(linewidth=3)
myv.show()
