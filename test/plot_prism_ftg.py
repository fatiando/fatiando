# -*- coding: utf-8 -*-
import sys
sys.path.append('../python/extmods')

import prismgrav as pg
import pylab as pl
import numpy as np
import time
import math

latmin, latmax = -250000, 250000
lonmin, lonmax = -250000, 250000
lats = np.arange(latmin, latmax, 5000, 'double')
lons = np.arange(lonmin, lonmax, 5000, 'double')
print "Grid size: %d x %d = %d" %(len(lons), len(lats), len(lons)*len(lats))

print "Calculating the field with c coded extention"
fields = []
calculators = [pg.prism_gxx, pg.prism_gxy, pg.prism_gxz, \
                             pg.prism_gyy, pg.prism_gyz, \
                                           pg.prism_gzz]

titles = ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
for title, calc in zip(*[titles, calculators]):
    start = time.clock()
    g = []
    for lat in lats:
        tmp = []
        for lon in lons:
            tmp.append(calc(2800, -100000,100000,-100000,100000,0,10000, lon, lat, -50000))
        g.append(tmp)
    finish = time.clock()
    print "%s time: %lf s" % (title, finish-start)
    fields.append(g)

print "Plotting"
glons, glats = pl.meshgrid(lons, lats)

# Plot the GGT
fig = pl.figure(figsize=(14,9))
pl.subplots_adjust(wspace=0.35)#hspace=0.15,

titles = []
titles.append(r"$g_{xx}$")
titles.append(r"$g_{xy}$")
titles.append(r"$g_{xz}$")
titles.append(r"$g_{yy}$")
titles.append(r"$g_{yz}$")
titles.append(r"$g_{zz}$")
for field, t in zip(*[fields, range(len(titles))]):
    ax = pl.subplot(2, 3, t+1, aspect='equal')
    pl.title(titles[t], fontsize=18)

    # Plot it
    pl.pcolor(glons, glats, field, cmap=pl.cm.jet)
    cb = pl.colorbar(orientation='vertical', format='%g', shrink=0.73)
    cb.set_label("Eotvos")
    pl.xlim(lonmin, lonmax)
    pl.ylim(latmin, latmax)

pl.show()