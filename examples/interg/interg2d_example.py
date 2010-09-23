"""
Invert a gravity profile for the relief of an interface.
"""

import pickle
import logging
log = logging.getLogger()
shandler = logging.StreamHandler()
shandler.setFormatter(logging.Formatter())
log.addHandler(shandler)
fhandler = logging.FileHandler("interg2d_example.log", 'w')
fhandler.setFormatter(logging.Formatter())
log.addHandler(fhandler)
log.setLevel(logging.DEBUG)

import numpy
import pylab

from fatiando.inversion import interg2d
from fatiando.gravity import io
import fatiando.geometry
import fatiando.utils
import fatiando.vis


