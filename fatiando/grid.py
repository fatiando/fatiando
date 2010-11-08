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
Create and operate on grids and profiles.

Functions:

    
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 26-Oct-2010'


import logging

import numpy

import fatiando


# Add the default handler (a null handler) to the logger to ensure that
# it won't print verbose if the program calling them doesn't want it
log = logging.getLogger('fatiando.grid')       
log.setLevel(logging.DEBUG)
log.addHandler(fatiando.default_log_handler)
