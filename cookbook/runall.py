# Copyright 2011 The Fatiando a Terra Development Team
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
