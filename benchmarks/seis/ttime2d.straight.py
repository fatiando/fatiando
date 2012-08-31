"""
Benchmark for fatiando.seis.ttime2d.straight
"""
import timeit
import fatiando as ft

setup = """
import numpy
import fatiando as ft
area = (0, 5, 0, 5)
shape = (30, 30)
model = ft.msh.dd.SquareMesh(area, shape)
model.addprop('vp', numpy.ones(model.size))
src_loc = ft.utils.random_points(area, 40)
rec_loc = ft.utils.circular_points(area, 40, random=True)
srcs, recs = ft.utils.connect_points(src_loc, rec_loc)

"""
n = 3
time = timeit.timeit("ft.seis.ttime2d.straight(model, 'vp', srcs, recs)",
                     setup, number=n)/float(n)
print "Average time of %d runs: %s" % (n, ft.utils.sec2hms(time))

