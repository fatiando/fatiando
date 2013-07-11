#!/usr/bin/python
# -*- coding: utf-8 -*-

from psegy import Segy
import numpy as np

with Segy("sub_vol.sgy") as sgy:
    ntrac = sgy.numbertraces
    nsamples = sgy.samplestrace
    
    trace = np.zeros([ntrac, nsamples])
    
    for i in range(ntrac):
        trace[i] = sgy.Traces(i).samples
    
    pass