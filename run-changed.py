"""
Run a list of recipes that where added or modified (given from hg st).

Pipe the output of "hg st cookbook" to this script.
"""
import os
import sys

for line in sys.stdin.readlines():
    status = line[0]
    if status == 'R':
        continue
    recipe = line[2:].strip()
    if recipe == 'cookbook/seismic_srtomo_sparse.py' or recipe[-3:] != '.py':
        continue
    module = recipe[9:-3]
    print "RUNNING: %s (%s)" % (recipe, module)
    print "==================================================================="
    rc = os.system('python %s' % (recipe))
    print "Return code:", rc
    if rc != 0:
        sys.exit()


