"""
Example of running a 2D straight-ray tomography on synthetic data generated
based on an image file. Since the image is big, use sparse matrices and steepest
descent (doesn't require Hessians).
WARNING: takes a long time to calculate (like 5 min)
"""
import time
from os import path
import numpy
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

imgfile = path.join(path.dirname(path.abspath(__file__)), 'fat-logo.png')
area = (0, 5, 0, 5)
shape = (150, 150)
model = ft.msh.dd.SquareMesh(area, shape)
model.img2prop(imgfile, 4, 10, 'vp')

log.info("Generating synthetic travel-time data")
src_loc = ft.utils.random_points(area, 200)
rec_loc = ft.utils.circular_points(area, 80, random=True)
srcs, recs = ft.utils.connect_points(src_loc, rec_loc)
start = time.time()
ttimes = ft.seis.ttime2d.straight(model, 'vp', srcs, recs, par=True)
log.info("  time: %s" % (ft.utils.sec2hms(time.time() - start)))
ttimes = ft.utils.contaminate(ttimes, 0.01, percent=True)

mesh = ft.msh.dd.SquareMesh(area, shape)
results = ft.seis.srtomo.run(ttimes, srcs, recs, mesh, sparse=True,
                             damping=0.05)
estimate, residuals = results
mesh.addprop('vp', estimate)

ft.vis.figure(figsize=(14, 5))
ft.vis.subplot(1, 2, 1)
ft.vis.axis('scaled')
ft.vis.title('Vp synthetic model of the Earth')
ft.vis.squaremesh(model, prop='vp', vmin=4, vmax=10, cmap=ft.vis.cm.seismic)
cb = ft.vis.colorbar()
cb.set_label('Velocity')
ft.vis.points(src_loc, '*y', label="Sources")
ft.vis.points(rec_loc, '^r', label="Receivers")
ft.vis.legend(loc='lower left', shadow=True, numpoints=1, prop={'size':10})
ft.vis.subplot(1, 2, 2)
ft.vis.axis('scaled')
ft.vis.title('Tomography result')
ft.vis.squaremesh(mesh, prop='vp', vmin=0.1, vmax=0.25,
    cmap=ft.vis.cm.seismic_r)
cb = ft.vis.colorbar()
cb.set_label('Slowness')
ft.vis.show()
