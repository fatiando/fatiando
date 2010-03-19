# -*- coding: utf-8 -*-
################################################################################
"""
SensiMat:
    Functions for constructing sensibility matrices.
"""
################################################################################
# Created on 08-Mar-2010
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__revision__ = '$Revision: 16 $'
__date__ = '$Date: 2010-03-19 14:30:43 -0300 (Fri, 19 Mar 2010) $'
################################################################################


def linear_grav(gfunc, modelspace, datax, datay, dataz):
    """
    Creates a sensibility matrix for a linear gravitational potential field (or
    derivatives) problem.

    Parameters:

        gfunc: callable object that calculated a field in question. Function
               parameters should be:
                   (density, w, e, s, n, top, bottom, x, y, z)
                   x, y, z are the coordinates of the observation point.

        modelspace: a list of dictionaries describing the discretized model
                    space. Each dictionary contains the information on a
                    specific geometric element (prism, tesseroid, etc.) and
                    should have the following keys:
                        'w', 'e', 's', 'n', 'top', 'bottom', 'dens'

        datax, datay, dataz: 1D lists with the x, y and z coordinates of each
                             data point.

    Return:

        A 2D list with the sensibility matrix
    """

    # A is gonna be the sensibility matrix that will be filled in the for loop
    A = []
    for i in range(len(datax)):

        # A tmp line of A
        line = []

        # Iterate over the model space to create a line of A
        for element in modelspace:

            line.append(gfunc(1, \
                              float(element['w']), float(element['e']), \
                              float(element['s']), float(element['n']), \
                              float(element['top']), float(element['bottom']), \
                              float(datax[i]), float(datay[i]), \
                              float(dataz[i])))

        # Add the line to A
        A.append(line)

    return A
