"""
Benchmark the vremove function of fatiando.msh.ddd
"""
import timeit
import fatiando as ft

setup = """
import fatiando as ft
import numpy as np
mesh = ft.msh.ddd.PrismMesh((0, 1, 0, 1, 0, 1), (50, 50, 50))
dens = np.zeros(mesh.size)
dens[10] = 1
mesh.addprop('density', dens)
"""
n = 20
print "Average time of %d runs:" % (n)
ctime = timeit.timeit("ft.msh.ddd.vremove(0, 'density', mesh)", setup, number=n)/float(n)
print ft.utils.sec2hms(ctime)
