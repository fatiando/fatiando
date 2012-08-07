"""
Run all the recipes
"""
import os
from subprocess import call
import re

basedir = 'recipes'
isrec = re.compile("\.py$", re.IGNORECASE).search
recipes = [os.path.sep.join([basedir, f]) for f in os.listdir(basedir) if isrec(f)]
for rec in recipes:
    print '\nRUNNING: %s' % (rec)
    call(['python', rec])
