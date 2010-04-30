"""
DataMani:
    Functions for generating synthetic gravity field data and contaminating it
    with random errors.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 07-Mar-2010'

import numpy

################################################################################
def contaminate(data, stddev, percent=True, return_stddev=False):
    """
    Contaminate a given data array (1D) with a normally distributed error of
    standard deviation stddev. If percent=True, then stddev is assumed to be a
    percentage of the maximum value in data (0 < stddev <= 1).
    If return_stddev=True, the calculated stddev will be returned with the 
    contaminated data.
    """

    # If stddev is a percentage, find the max and calculate the percentage of it
    if percent:

        # Convert data into a numpy.array to make it easier to find the maximum
        maximum = abs(numpy.array(data)).max()

        # Calculate a percentage of it that will be the new stddev
        stddev = stddev*maximum

    # Contaminate all the data in data and store it in a new list
    cont_data = []
    for data in data:

        # Append the new data belonging to a normal distribution with the old
        # data as mean and stddev standard deviation
        cont_data.append(numpy.random.normal(data, stddev))

    # Return the new data
    if return_stddev:
        
        return [cont_data, stddev]
    
    else:
        
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