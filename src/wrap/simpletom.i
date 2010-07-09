/* **************************************************************************
   Interface file for generation SWIG wrappers around 'simpletom.c'
   ************************************************************************** */

/* The module docstring */
%define DOCSTRING
"
seismo.simple:
    C-coded direct model for a simplified Cartesian tomography in which there is
    no reflection or refraction.

Author: Leonardo Uieda
Created 29 April 2010
"
%enddef

/* Declare the module name */
%module(docstring=DOCSTRING) simple

/* ************************************************************************** */

/* Put the headers with the definitions */
/* ************************************************************************** */
%{

#include "../c/directmodels/simpletom.h"

%}
/* ************************************************************************** */

/* OUT TYPEMAP FOR TRAVELTIME */
%typemap(out) double 
{    
    /* Check if the return was -1 (error) */
    if($1 == -1)
    {        
        /* Raise the appropriate exception */
        PyErr_SetString(PyExc_ValueError, "Raypath intercepted cell in more than two points. This can't be right. Check the traveltime function for errors.");
        return NULL;
    }
    
    $result = PyFloat_FromDouble($1);
}

/* Expose the functions and variables that will be wrapped */
/* ************************************************************************** */
/* ************************************************************************** */

/* TRAVELTIME */
/* ************************************************************************** */
%feature("autodoc", "1");
%define TRAVELTIMEDOC
"
Returns the travel time inside a given cell.

Parameters:

    slowness: the slowness of the cell. Must be in units compatible with
        the other parameters!

    x1, y1: coordinates of the lower-left corner of the cell

    x2, y2: coordinates of the upper-right corner of the cell

    x_src, y_src: coordinates of the wave source

    x_rec, y_rec: coordinates of the reciever
"
%enddef
%feature("docstring", TRAVELTIMEDOC);
extern double traveltime(double slowness, double x1, double y1,
                         double x2, double y2,
                         double x_src, double y_src,
                         double x_rec, double y_rec);
/* ************************************************************************** */

/* ************************************************************************** */
/* ************************************************************************** */
