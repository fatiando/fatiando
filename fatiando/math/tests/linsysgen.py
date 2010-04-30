"""
A little script to generate the linear systems and LU decompositions to use in
the testing of my lu decomp and lin sys solver functions. Uses numpy for ramdom
number generation and scipy.linalg for the LU decompositions.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 04-Mar-2010'

import numpy
import pylab
import scipy.linalg
import os

def mktestdata(datadir='testdata', verbose=False):

    if verbose:
        print "Generating test data..."

    if not os.path.exists(datadir):
        if verbose:
            print "Creating data directory '%s'..." % (datadir)
        os.mkdir(datadir)

    if verbose:
        print "Generating random linear systems..."
    num = 10
    orders = [2, 3, 10, 1000]
    ranges = [1, 1000]
    As = []
    ys = []
    for o in orders:
        for r in ranges:
            As.extend(numpy.random.uniform(-r,r,(num,o,o)).tolist())
            ys.extend(numpy.random.uniform(-r,r,(num,o)).tolist())

    if verbose:
        print "Generating LU decomps and solving the systems..."
    systems = zip(*[As, ys])
    sysnum = 0
    for A, y in systems:

        sysnum += 1

        LU, p = scipy.linalg.lu_factor(A)

        x = scipy.linalg.lu_solve((LU,p), y)

        # Save the results
        pylab.savetxt(os.path.join(datadir,'matrix%d.txt' % (sysnum)), A)
        pylab.savetxt(os.path.join(datadir,'data%d.txt' % (sysnum)), y)

        pylab.savetxt(os.path.join(datadir,'lu%d.txt' % (sysnum)), LU)
        pylab.savetxt(os.path.join(datadir,'permut%d.txt' % (sysnum)), p)

        pylab.savetxt(os.path.join(datadir,'solution%d.txt' % (sysnum)), x)

    if verbose:
        print "Total %d test cases" % (sysnum)


if __name__ == '__main__':
    mktestdata(verbose=True)