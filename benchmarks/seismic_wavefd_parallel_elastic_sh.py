"""
Benchmark the speed of seismic.wavefd.elastic_sh with varying number of jobs.
"""
import time
import numpy as np
from fatiando import seismic, logger, gridder, vis, utils

partitions = [(1,1), (2,1), (2,2), (2,3), (2,4), (2,5), (3,4), (4,4)]
njobs = [p[0]*p[1] for p in partitions]
times = []
for partition in partitions:
    jobs = partition[0]*partition[1]
    print 'Running with ', jobs, 'job(s)'
    sources = [seismic.wavefd.MexHatSource(25, 25, 100, 0.5, delay=1.5)]
    shape = (500, 500)
    spacing = (1000, 1000)
    area = (0, spacing[1]*shape[1], 0, spacing[0]*shape[0])
    dens = 2700*np.ones(shape)
    svel = 3000*np.ones(shape)
    dt = 0.05
    maxit = 500
    start = time.time()
    timesteps = seismic.wavefd.elastic_sh(spacing, shape, svel, dens, dt, maxit,
        sources, padding=0.5, partition=partition)
    for step in timesteps:
        continue
    times.append(time.time() - start)
    print '  time:', utils.sec2hms(times[-1])

print 'Plotting'
np.savetxt('seismic_wavefd_elastic_sh.dat', np.transpose([njobs, times]))
vis.mpl.figure()
vis.mpl.plot(njobs, times, '.-k')
vis.mpl.xlabel('Processes')
vis.mpl.ylabel('Time (seconds)')
vis.mpl.show()



