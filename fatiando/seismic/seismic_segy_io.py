r"""
Example using the psgy module.

Reads a segy file apply a moving average over the samples
and save as a new file. Booth are plotted side by side at end.

Uses matplotlib for all plotting stuff.

*Note: could include in the cookbook partial traces reading of segy (for huge files)
"""

from psegy import Segy, EBCDIC, SegyBH
import numpy as np
import matplotlib.pylab as pylab
from matplotlib.pyplot import cm


# Read the segy file and get the traces
with Segy("cloudpspin635.segy") as sgy:
    ntrac = sgy.numbertraces
    ns = sgy.samplestrace    
    cloudp_traces = []
    cloud_image_trace = np.zeros([ntrac, ns])
    
    for i in range(ntrac):
        cloud_image_trace[i] = sgy.traces(i).samples
        cloudp_traces.append(sgy.traces(i))
        
# apply moving average over trace samples size 5 samples
for trace in cloudp_traces:
    ns = trace.numbersamples()
    for i in range(ns):
        s2 = s1 = s_2 = s_1 = trace.samples[i]
        if i-2 > -1 : 
            s_2 = trace.samples[i-2]
        if i-1 > -1 : 
            s_1 = trace.samples[i-1]
        if i+2 < ns : 
            s2 = trace.samples[i+2]
        if i+1 < ns:            
            s1 = trace.samples[i+1]                        
        trace.samples[i] = (s1+s2+s_1+s_2+trace.samples[i])/5.0
        
# write back the new traces as a new file
segy = Segy("cloudpspin635_smooth.segy", cloudp_traces, EBCDIC, SegyBH())
segy.write()

# Read the newly created file for plotting 
with Segy("cloudpspin635_smooth.segy") as sgy:
    ntrac = sgy.numbertraces
    nsamples = sgy.samplestrace   
    cloud_smooth_image_trace = np.zeros([ntrac, nsamples])
    
    for i in range(ntrac):
        cloud_smooth_image_trace[i] = sgy.traces(i).samples


# that's default image size for this interactive session
pylab.rcParams['figure.figsize'] = 15, 15 

fig = pylab.figure()
ax = fig.add_subplot(1,2,1)
ax.imshow(cloud_image_trace.transpose(), interpolation='bicubic', cmap=cm.gray, vmin=-25, vmax=25, aspect=0.5, origin='upper', extent=[0,ntrac*10,0,nsamples*sgy.samplerate*0.001])
ax.set_title('Cloudpsin',fontsize=10)

ax = fig.add_subplot(1,2,2)
ax.imshow(cloud_smooth_image_trace.transpose(), interpolation='bicubic', cmap=cm.gray, vmin=-25, vmax=25, aspect=0.5, origin='upper', extent=[0,ntrac*10,0,nsamples*sgy.samplerate*0.001])
ax.set_title('Cloudpsin moving average',fontsize=10)

#pylab.rcParams['figure.figsize'] = 4, 4  # that's default image size for this interactive session