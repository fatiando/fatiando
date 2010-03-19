# -*- coding: utf-8 -*-
################################################################################
"""
DataGen:
    Functions for generating synthetic gravity field data and contaminating it
    with random errors.
"""
################################################################################
# Created on 07-Mar-2010
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__revision__ = '$Revision: 16 $'
__date__ = '$Date: 2010-03-19 14:30:43 -0300 (Fri, 19 Mar 2010) $'
################################################################################

import numpy

################################################################################
def grid_gdata(gfunc, model, gxs, gys, height):
    """
    Generate a regular grid of 'gfunc' that represents a function that
    calculates a specific gravity field component given a prism model.

    Parameters:

        gfunc: callable object that calculated a gravity field component
               function parameters should be:
                   (density, w, e, s, n, top, bottom, x, y, z)
                   x, y, z are the coordinates of the observation point.

        model: a list of dictionaries containing information on the model
               elements. Each dict should have the keys:
                  'w', 'e', 's', 'n', 'top', 'bottom', 'dens'

        gxs: 1D list with the x (longitude) values of the grid nodes

        gys: 1D list with the y (latitude) values of the grid nodes

        height: height in which the field will be calculated

    Return:

        A 1D list with the gridded data
    """

    # Create the data
    gdata = []
    for i in range(len(gxs)):

        # Iterate over the model
        tmpval = 0
        for element in model:

            tmpval += gfunc(element['dens'], \
                element['w'], element['e'], element['s'], element['n'], \
                element['top'], element['bottom'], \
                float(gxs[i]), float(gys[i]), height)

        # Add to the grid data
        gdata.append(tmpval)

    # Return the grid
    return gdata


def contaminate(gdata, stddev, percent=True):
    """
    Contaminate a given data array (1D) with a normally distributed error of
    standard deviation stddev. If percent=True, then stddev is assumed to be a
    percentage of the maximum value in gdata (0 < stddev <= 1).
    """

    # If stddev is a percentage, find the max and calculate the percentage of it
    if percent:

        # Convert gdata into a numpy.array to make it easier to find the maximum
        maximum = abs(numpy.array(gdata)).max()

        # Calculate a percentage of it that will be the new stddev
        stddev = stddev*maximum

    # Contaminate all the data in gdata and store it in a new list
    cont_data = []
    for data in gdata:

        # Append the new data belonging to a normal distribution with the old
        # data as mean and stddev standard deviation
        cont_data.append(numpy.random.normal(data, stddev))

    # Return the new data
    return cont_data

################################################################################

if __name__ == '__main__':

    print "Plotting sample data and contaminated data"

    import pylab
    from extmods import prismgrav

    lons = numpy.arange(-2000,2000,100)
    lats = numpy.arange(-2000,2000,100)
    glons, glats = pylab.meshgrid(lons, lats)

    model = []
    prism = {'w':-500, 'e':500, 's':-1000, 'n':1000, 'top':200, 'bottom':800,
             'dens':1000}
    model.append(prism)

    gdata = grid_gdata(prismgrav.prism_gxy, model, \
                       glons.ravel(), glats.ravel(), 0)

    percent = 0.02
    cont = contaminate(gdata, percent)

    cont = [cont[i:i+len(lons)] for i in range(0, len(cont), len(lons))]
    gdata = [gdata[i:i+len(lons)] for i in range(0, len(gdata), len(lons))]

    # Plot
    pylab.figure(figsize=(14,10))
    pylab.subplot(1, 2, 1)
    pylab.title("Sample data")
    cs = pylab.contour(glons, glats, gdata, 15, colors='k', linestyles='solid')
    pylab.clabel(cs, fmt='%g')
    #pylab.pcolor(glons, glats, gdata, cmap=pylab.cm.jet)
    pylab.xlim(lons[0], lons[-1])
    pylab.ylim(lats[0], lats[-1])
    pylab.xlabel('x')
    pylab.ylabel('y')

    pylab.subplot(1, 2, 2)
    pylab.title("Contaminated with %g percent" % (percent))
    cont_cs = pylab.contour(glons, glats, cont, 15, colors='k', linestyles='solid')
    pylab.clabel(cont_cs, fmt='%g')
    #pylab.pcolor(glons, glats, gdata, cmap=pylab.cm.jet)
    pylab.xlim(lons[0], lons[-1])
    pylab.ylim(lats[0], lats[-1])
    pylab.xlabel('x')

    pylab.show()