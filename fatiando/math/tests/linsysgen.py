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
        
        print "Generating random linear systems,LU decomps and inverses..."
        
        
    orders = range(2,10,1)
    orders.extend([10, 100, 200])
    
    ranges = numpy.ones(10)
    
    sysnum = 0
    
    for o in orders:
        
        for r in ranges:

            sysnum += 1
            
            A = numpy.random.uniform(-r,r,(o,o)).tolist()
            
            y = numpy.random.uniform(-r,r,(o)).tolist()            

            LU, p = scipy.linalg.lu_factor(A)
    
            x = scipy.linalg.lu_solve((LU,p), y)
            
            inv = scipy.linalg.inv(A)
    
            # Save the results
            pylab.savetxt(os.path.join(datadir,'matrix%d.txt' % (sysnum)), A)
            pylab.savetxt(os.path.join(datadir,'data%d.txt' % (sysnum)), y)
    
            pylab.savetxt(os.path.join(datadir,'lu%d.txt' % (sysnum)), LU)
            pylab.savetxt(os.path.join(datadir,'permut%d.txt' % (sysnum)), p)
    
            pylab.savetxt(os.path.join(datadir,'solution%d.txt' % (sysnum)), x)
            
            pylab.savetxt(os.path.join(datadir,'inverse%d.txt' % (sysnum)), inv)       

    if verbose:
        
        print "Total %d test cases" % (sysnum)


if __name__ == '__main__':
    
    mktestdata(verbose=True)