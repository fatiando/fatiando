# -*- coding: utf-8 -*-
################################################################################
"""
A little script to generate the linear systems and LU decompositions to use in
the testing of my lu decomp and lin sys solver functions. Uses numpy for ramdom
number generation and scipy.linalg for the LU decompositions.
"""
################################################################################
# Created on 04-Mar-2010
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__revision__ = '$Revision: 15 $'
__date__ = '$Date: 2010-03-17 14:06:42 -0300 (Wed, 17 Mar 2010) $'
################################################################################

import numpy
import pylab
import scipy.linalg

print "Generating random linear systems..."
num = 5
orders = [2, 3, 10, 1000]
ranges = [1, 1000]
As = []
ys = []
for o in orders:
    for r in ranges:
        As.extend(numpy.random.uniform(-r,r,(num,o,o)).tolist())
        ys.extend(numpy.random.uniform(-r,r,(num,o)).tolist())

print "Generating LU decomps and solving the systems..."
systems = zip(*[As, ys])
sysnum = 0
for A, y in systems:

    sysnum += 1

    LU, p = scipy.linalg.lu_factor(A)

    x = scipy.linalg.lu_solve((LU,p), y)

    # Save the results
    pylab.savetxt('linsys-data/matrix%d.txt' % (sysnum), A)
    pylab.savetxt('linsys-data/data%d.txt' % (sysnum), y)

    pylab.savetxt('linsys-data/lu%d.txt' % (sysnum), LU)
    pylab.savetxt('linsys-data/permut%d.txt' % (sysnum), p)

    pylab.savetxt('linsys-data/solution%d.txt' % (sysnum), x)

print "Total %d test cases" % (sysnum)