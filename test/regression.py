#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Regression testing framework.

This module will search for scripts in the same directory named *_test.py. Each
such script should be a test suite that tests a module through PyUnit (as of
Python 2.1, PyUnit is included in the standard library as 'unittest'). This
script will aggregate all found test suites into one big test suite and run them
all at once.

This program is a modified version of 'regression.py' that is part of
"Dive Into Python", a free Python book for experienced programmers by Mark
Pilgrim. Visit http://diveintopython.org/ for the latest version.
"""

import sys
import os
import re
import unittest

def regressionTest():
    path = os.path.abspath(os.path.dirname(sys.argv[0]))
    files = os.listdir(path)
    test = re.compile("(^test\_).+(\.py$)", re.IGNORECASE)
    files = filter(test.search, files)
    filenameToModuleName = lambda f: os.path.splitext(f)[0]
    moduleNames = map(filenameToModuleName, files)
    modules = map(__import__, moduleNames)
    load = unittest.defaultTestLoader.loadTestsFromModule
    return unittest.TestSuite(map(load, modules))

if __name__ == "__main__":
    unittest.main(defaultTest="regressionTest")
