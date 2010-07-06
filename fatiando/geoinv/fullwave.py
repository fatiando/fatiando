"""
FullWave:
    Full waveform inversion using Finite Difference solvers.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 05-Jul-2010'

import time
import logging
import math

import pylab
import numpy

import fatiando
from fatiando.geoinv.lmsolver import LMSolver


logger = logging.getLogger('FullWave')       
logger.setLevel(logging.DEBUG)
logger.addHandler(fatiando.default_log_handler)