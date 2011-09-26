# Copyright 2010 The Fatiando a Terra Development Team
#
# This file is part of Fatiando a Terra.
#
# Fatiando a Terra is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Fatiando a Terra is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Fatiando a Terra.  If not, see <http://www.gnu.org/licenses/>.
"""
Run all tests in the fatiando.tests package

Based on the regression test program found in the book Dive into Python by
Mark Pilgrim.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 26-Sep-2011'


import sys
import os
import re
import unittest

def find_tests(basepath):
    """
    Finds all test modules in basepath whose names end with _test.py
    """
    istest = re.compile("_test\.py$", re.IGNORECASE).search
    files = [f for f in os.listdir(basepath) if istest(f)]
    modules = [os.path.splitext(f)[0] for f in files]
    return ['.'.join(['fatiando.tests', m]) for m in modules]

def import_tests(tests):
    """
    Returns the imported test modules given a list of file names.
    """
    # import returns the base package. so need to look for modules in sys
    tmp = [__import__(m) for m in tests]
    return [sys.modules[m] for m in tests]

def testsuite():
    """
    Returns a test suite
    """
    modules = import_tests(find_tests(os.path.join('fatiando','tests')))
    print "Found %d tests\n" % (len(modules))
    load = unittest.defaultTestLoader.loadTestsFromModule
    return unittest.TestSuite(map(load, modules))

if __name__ == "__main__":
    unittest.main(defaultTest="testsuite")
