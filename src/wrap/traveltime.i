/* *****************************************************************************
 Copyright 2010 Leonardo Uieda

 This file is part of Fatiando a Terra.

 Fatiando a Terra is free software: you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 Fatiando a Terra is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public License
 along with Fatiando a Terra.  If not, see <http://www.gnu.org/licenses/>.
 **************************************************************************** */
 
/* **************************************************************************
   Interface file for generation SWIG wrappers around 'traveltime.c'
   ************************************************************************** */

/* The module docstring */
%define DOCSTRING
"
TravelTime:
    C-coded functions for calculating the travel times of seismic waves.

Author: Leonardo Uieda
Created 29 April 2010
"
%enddef

/* Declare the module name */
%module(docstring=DOCSTRING) traveltime

%{

#include "../c/traveltime.h"

%}

/* OUT TYPEMAP FOR cartesian_straight */
%typemap(out) double 
{    
    /* Check if the return was -1 (error) */
    if($1 == -1)
    {        
        /* Raise the appropriate exception */
        PyErr_SetString(PyExc_ValueError, "Raypath intercepted cell in more than two points");
        return NULL;
    }
    
    $result = PyFloat_FromDouble($1);
}

/* cartesian_straight */
/* ************************************************************************** */
%feature("autodoc", "1");
%define CART_STRAIGHT_DOC
"
Calculate the travel time inside a square cell assuming the ray is a straight
line.

Parameters:

    slowness: the slowness of the cell. Must be in units compatible with
        the other parameters!

    x1, y1: coordinates of the lower-left corner of the cell

    x2, y2: coordinates of the upper-right corner of the cell

    x_src, y_src: coordinates of the wave source

    x_rec, y_rec: coordinates of the receiver
"
%enddef
%feature("docstring", CART_STRAIGHT_DOC);
extern double cartesian_straight(double slowness, double x1, double y1,
                                 double x2, double y2,
                                 double x_src, double y_src,
                                 double x_rec, double y_rec);