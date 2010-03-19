/* **************************************************************************
   Interface file for generation SWIG wrappers around 'prismgrav.c'
   ************************************************************************** */

/* The module docstring */
%define DOCSTRING
"C-coded extention module that contains a set of functions that calculates the gravitational potential and its first and second derivatives for the rectangular prism using the formulas in Nagy (2000).

Author: Leonardo Uieda
Date: 01 March 2010
Last Update: $DATE: $
$REVISION: $
"
%enddef

/* Declare the module name */
%module(docstring=DOCSTRING) prismgrav

/* ************************************************************************** */

/* Put the headers with the definitions */
/* ************************************************************************** */
%{

#include "prismgrav.h"

%}
/* ************************************************************************** */

/* Expose the functions and variables that will be wrapped */
/* ************************************************************************** */
/* ************************************************************************** */

/* GXX */
/* ************************************************************************** */
%feature("autodoc", "1");
%define PGXXDOC
"
Calculates the gravity gradient tensor component gxx caused by a prism using the formulas given in Nagy (2000). The coordinate system of the input parameters is supposed to be x->north, y->east; z->down.
Accepts values in SI units(!) and returns values in Eotvos!
Parameters:
    * double dens: density of the prism;
    * double x1, x2, y1, ... z2: the borders of the prism;
    * double xp, yp, zp: coordinates of the point P where the effect will be calculated;
"
%enddef
%feature("docstring", PGXXDOC);
extern double prism_gxx(double dens, double x1, double x2, double y1, double y2,
                        double z1, double z2, double xp, double yp, double zp);
/* ************************************************************************** */

/* GXY */
/* ************************************************************************** */
%feature("autodoc", "1");
%define PGXYDOC
"
Calculates the gravity gradient tensor component gxy caused by a prism using the formulas given in Nagy (2000). The coordinate system of the input parameters is supposed to be x->north, y->east; z->down.
Accepts values in SI units(!) and returns values in Eotvos!
Parameters:
    * double dens: density of the prism;
    * double x1, x2, y1, ... z2: the borders of the prism;
    * double xp, yp, zp: coordinates of the point P where the effect will be calculated;
"
%enddef
%feature("docstring", PGXYDOC);
extern double prism_gxy(double dens, double x1, double x2, double y1, double y2,
                        double z1, double z2, double xp, double yp, double zp);
/* ************************************************************************** */

/* GXZ */
/* ************************************************************************** */
%feature("autodoc", "1");
%define PGXZDOC
"
Calculates the gravity gradient tensor component gxz caused by a prism using the formulas given in Nagy (2000). The coordinate system of the input parameters is supposed to be x->north, y->east; z->down.
Accepts values in SI units(!) and returns values in Eotvos!
Parameters:
    * double dens: density of the prism;
    * double x1, x2, y1, ... z2: the borders of the prism;
    * double xp, yp, zp: coordinates of the point P where the effect will be calculated;
"
%enddef
%feature("docstring", PGXZDOC);
extern double prism_gxz(double dens, double x1, double x2, double y1, double y2,
                        double z1, double z2, double xp, double yp, double zp);
/* ************************************************************************** */

/* GYY */
/* ************************************************************************** */
%feature("autodoc", "1");
%define PGYYDOC
"
Calculates the gravity gradient tensor component gyy caused by a prism using the formulas given in Nagy (2000). The coordinate system of the input parameters is supposed to be x->north, y->east; z->down.
Accepts values in SI units(!) and returns values in Eotvos!
Parameters:
    * double dens: density of the prism;
    * double x1, x2, y1, ... z2: the borders of the prism;
    * double xp, yp, zp: coordinates of the point P where the effect will be calculated;
"
%enddef
%feature("docstring", PGYYDOC);
extern double prism_gyy(double dens, double x1, double x2, double y1, double y2,
                        double z1, double z2, double xp, double yp, double zp);
/* ************************************************************************** */

/* GYZ */
/* ************************************************************************** */
%feature("autodoc", "1");
%define PGYZDOC
"
Calculates the gravity gradient tensor component gyz caused by a prism using the formulas given in Nagy (2000). The coordinate system of the input parameters is supposed to be x->north, y->east; z->down.
Accepts values in SI units(!) and returns values in Eotvos!
Parameters:
    * double dens: density of the prism;
    * double x1, x2, y1, ... z2: the borders of the prism;
    * double xp, yp, zp: coordinates of the point P where the effect will be calculated;
"
%enddef
%feature("docstring", PGYZDOC);
extern double prism_gyz(double dens, double x1, double x2, double y1, double y2,
                        double z1, double z2, double xp, double yp, double zp);
/* ************************************************************************** */

/* GZZ */
/* ************************************************************************** */
%feature("autodoc", "1");
%define PGZZDOC
"
Calculates the gravity gradient tensor component gzz caused by a prism using the formulas given in Nagy (2000). The coordinate system of the input parameters is supposed to be x->north, y->east; z->down.
Accepts values in SI units(!) and returns values in Eotvos!
Parameters:
    * double dens: density of the prism;
    * double x1, x2, y1, ... z2: the borders of the prism;
    * double xp, yp, zp: coordinates of the point P where the effect will be calculated;
"
%enddef
%feature("docstring", PGZZDOC);
extern double prism_gzz(double dens, double x1, double x2, double y1, double y2,
                        double z1, double z2, double xp, double yp, double zp);
/* ************************************************************************** */


/* ************************************************************************** */
/* ************************************************************************** */