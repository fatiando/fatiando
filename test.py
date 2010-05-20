"""
Run the Fatiando test suite.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 28-Mar-2010'


import sys

import fatiando


def run(args=[]):

    verbosity = False
    
    label = 'fast'

    for arg in args[1:]:
        
        if arg == '-v':
            
            verbosity = True
            
        elif arg == '-full':
            
            label = 'full'
            
        elif arg == '-h':     
                               
            helpmsg = \
            """
            Test runner for the Fatiando test suite.
            
            python test.py [options]
            
            Options:
            
                -v: activate verbosity
            
                -full: Run the full test suite. The default is running a smaller
                       version of the test suite (better when running the tests 
                       multiple times)
            """
            
            print helpmsg
            
        else:
            
            print "Invalid option. Use 'python test.py -h' for help."
            
            return 0

    fatiando.test(label=label, verbose=verbosity)


if __name__ == '__main__':

    run(sys.argv)