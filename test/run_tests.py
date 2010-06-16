#!/usr/bin/env python
"""
Run the integration tests.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 16-Jun-2010'

import os
import sys
import subprocess
import time

def main():
    
    start = time.time()
    
    sys.stderr.write("*** Integration test suite ***\n\n")

    mydir = os.path.abspath(os.path.dirname(__file__))
    
    files = os.listdir(mydir)

    os.chdir(mydir)
                                                                        
    procs = []

    for fname in files:

        full_name = os.path.join(mydir, fname)

        if not os.path.isdir(full_name) \
            and os.path.splitext(fname)[-1] == '.py' \
            and not os.path.samefile(os.path.abspath(fname), \
                        os.path.abspath(os.path.split(__file__)[-1])):

            sys.stderr.write("Running %s\n" % (fname))

            procs.append(subprocess.Popen(['python', fname]))

    for proc in procs:

        proc.wait()

    end = time.time()

    print "Ran %d tests in %g seconds" % (len(procs), end - start)
    
    
if __name__ == '__main__':
    
    main()