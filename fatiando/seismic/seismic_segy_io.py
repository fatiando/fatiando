r"""
Example using the psgy module to read a segy file
and create a matrix of traces by samples.
2D seismic case example

include in the cookbook partial traces reading of segy (for huge files)
site at lest

put write and read in cookbook.


"""

from psegy import Segy, EBCDIC, Trace, SegyTH, SegyBH
import numpy as np

with Segy("sub_vol.sgy") as sgy:
    ntrac = sgy.numbertraces
    nsamples = sgy.samplestrace
    
    trace = np.zeros([ntrac, nsamples], np.float32)
    
    for i in range(ntrac):
        trace[i] = sgy.traces(i).samples
    
    pass


traces = []
with Segy("sub_vol.sgy") as sgy:
    ntrac = sgy.numbertraces
    nsamples = sgy.samplestrace
    
    for i in range(ntrac):
        traces.append(sgy.traces(i))
    
segy = Segy("sub_vol_fat.sgy", traces, EBCDIC, SegyBH())
segy.write()