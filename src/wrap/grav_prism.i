/* *****************************************************************************
 Copyright 2010 The Fatiando a Terra Development Team

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
   Interface file for generation SWIG wrappers around 'grav_prism.c'
   ************************************************************************** */

/* The module docstring */
%define DOCSTRING
"Functions to calculate the gravitational potential and its first and second 
derivatives for the right rectangular prism using the formulas in Nagy et al.
(2000).
"
%enddef

/* Declare the module name */
%module(docstring=DOCSTRING) prism

%{

#include "../c/grav_prism.h"

%}

/* GZ */
/* ************************************************************************** */
%feature("autodoc", "1");
%define PGZDOC
"
Calculates the gz gravity component caused by a right prism using the formulas 
given in Nagy (2000). The coordinate system of the input parameters is supposed 
to be x->north, y->east; z->down.
Accepts values in SI units(!) and returns values in Eotvos!
Parameters:
  * double dens: density of the prism;
  * double x1, x2, y1, ... z2: the borders of the prism;
  * double xp, yp, zp: coordinates of the point P where the effect will be 
    calculated;
"
%enddef
%feature("docstring", PGZDOC);
%rename(gz) prism_gz;
extern double prism_gz(double dens, double x1, double x2, double y1, double y2,
                       double z1, double z2, double xp, double yp, double zp);
/* ************************************************************************** */

/* GXX */
/* ************************************************************************** */
%feature("autodoc", "1");
%define PGXXDOC
"
Calculates the gravity gradient tensor component gxx caused by a prism using the
formulas given in Nagy (2000). The coordinate system of the input parameters is 
supposed to be x->north, y->east; z->down.
Accepts values in SI units(!) and returns values in Eotvos!
Parameters:
  * double dens: density of the prism;
  * double x1, x2, y1, ... z2: the borders of the prism;
  * double xp, yp, zp: coordinates of the point P where the effect will be 
    calculated;
"
%enddef
%feature("docstring", PGXXDOC);
%rename(gxx) prism_gxx;
extern double prism_gxx(double dens, double x1, double x2, double y1, double y2,
                        double z1, double z2, double xp, double yp, double zp);
/* ************************************************************************** */

/* GXY */
/* ************************************************************************** */
%feature("autodoc", "1");
%define PGXYDOC
"
Calculates the gravity gradient tensor component gxy caused by a prism using the
formulas given in Nagy (2000). The coordinate system of the input parameters is
supposed to be x->north, y->east; z->down.
Accepts values in SI units(!) and returns values in Eotvos!
Parameters:
  * double dens: density of the prism;
  * double x1, x2, y1, ... z2: the borders of the prism;
  * double xp, yp, zp: coordinates of the point P where the effect will be 
    calculated;
"
%enddef
%feature("docstring", PGXYDOC);
%rename(gxy) prism_gxy;
extern double prism_gxy(double dens, double x1, double x2, double y1, double y2,
                        double z1, double z2, double xp, double yp, double zp);
/* ************************************************************************** */

/* GXZ */
/* ************************************************************************** */
%feature("autodoc", "1");
%define PGXZDOC
"
Calculates the gravity gradient tensor component gxz caused by a prism using the
formulas given in Nagy (2000). The coordinate system of the input parameters is
supposed to be x->north, y->east; z->down.
Accepts values in SI units(!) and returns values in Eotvos!
Parameters:
  * double dens: density of the prism;
  * double x1, x2, y1, ... z2: the borders of the prism;
  * double xp, yp, zp: coordinates of the point P where the effect will be 
    calculated;
"
%enddef
%feature("docstring", PGXZDOC);
%rename(gxz) prism_gxz;
extern double prism_gxz(double dens, double x1, double x2, double y1, double y2,
                        double z1, double z2, double xp, double yp, double zp);
/* ************************************************************************** */

/* GYY */
/* ************************************************************************** */
%feature("autodoc", "1");
%define PGYYDOC
"
Calculates the gravity gradient tensor component gyy caused by a prism using the
formulas given in Nagy (2000). The coordinate system of the input parameters is
supposed to be x->north, y->east; z->down.
Accepts values in SI units(!) and returns values in Eotvos!
Parameters:
  * double dens: density of the prism;
  * double x1, x2, y1, ... z2: the borders of the prism;
  * double xp, yp, zp: coordinates of the point P where the effect will be 
    calculated;
"
%enddef
%feature("docstring", PGYYDOC);
%rename(gyy) prism_gyy;
extern double prism_gyy(double dens, double x1, double x2, double y1, double y2,
                        double z1, double z2, double xp, double yp, double zp);
/* ************************************************************************** */

/* GYZ */
/* ************************************************************************** */
%feature("autodoc", "1");
%define PGYZDOC
"
Calculates the gravity gradient tensor component gyz caused by a prism using the
formulas given in Nagy (2000). The coordinate system of the input parameters is
supposed to be x->north, y->east; z->down.
Accepts values in SI units(!) and returns values in Eotvos!
Parameters:
  * double dens: density of the prism;
  * double x1, x2, y1, ... z2: the borders of the prism;
  * double xp, yp, zp: coordinates of the point P where the effect will be 
    calculated;
"
%enddef
%feature("docstring", PGYZDOC);
%rename(gyz) prism_gyz;
extern double prism_gyz(double dens, double x1, double x2, double y1, double y2,
                        double z1, double z2, double xp, double yp, double zp);
/* ************************************************************************** */

/* GZZ */
/* ************************************************************************** */
%feature("autodoc", "1");
%define PGZZDOC
"
Calculates the gravity gradient tensor component gzz caused by a prism using the
formulas given in Nagy (2000). The coordinate system of the input parameters is
supposed to be x->north, y->east; z->down.
Accepts values in SI units(!) and returns values in Eotvos!
Parameters:
  * double dens: density of the prism;
  * double x1, x2, y1, ... z2: the borders of the prism;
  * double xp, yp, zp: coordinates of the point P where the effect will be
    calculated;
"
%enddef
%feature("docstring", PGZZDOC);
%rename(gzz) prism_gzz;
extern double prism_gzz(double dens, double x1, double x2, double y1, double y2,
                        double z1, double z2, double xp, double yp, double zp);
/* ************************************************************************** */