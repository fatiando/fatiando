"""
Contaminate:
    Functions for contaminating data arrays with various kinds of random errors.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 07-Mar-2010'


import numpy


def gaussian(data, stddev, percent=True, return_stddev=False):
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
    for value in data:

        # Append the new data belonging to a normal distribution with the old
        # data as mean and stddev standard deviation
        cont_data.append(numpy.random.normal(value, stddev))

    # Return the new data
    if return_stddev:
        
        return [cont_data, stddev]
    
    else:
        
        return cont_data

