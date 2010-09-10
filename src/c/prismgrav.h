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

   This module contains a set of functions that calculate the gravitational
   potential and its first and second derivatives for the rectangular prism
   using the formulas in Nagy (2000).

   Author: Leonardo Uieda
   Date: 01 March 2010

   ************************************************************************** */

#ifndef _PRISMGRAV_H_
#define _PRISMGRAV_H_

/* CONSTANTS */
/* ************************************************************************** */

/* The gravitational constant (m^3*kg^-1*s^-1) */
#define G 0.00000000006673

/* Conversion factor from SI units to Eotvos: 1 /s**2 = 10**9 Eotvos */
#define SI2EOTVOS 1000000000.0

/* Conversion factor from SI units to mGal: 1 m/s**2 = 10**5 mGal */
#define SI2MGAL 100000.0

/* ************************************************************************** */


/* FUNCTION DECLARATIONS */
/* ************************************************************************** */


/* Calculates the components of the gravity attraction caused by a right
rectangular prism using the formulas given in Nagy (2000).

The coordinate system of the input parameters is assumed to be
    x->north, y->east; z->down.

Input values in SI units and returns values in Eotvos!

Parameters:
    * double dens: density of the prism;
    * double x1, x2, y1, ... z2: the borders of the prism;
    * double xp, yp, zp: coordinates of the point P where the effect will be
                         calculated;
*/
extern double prism_gz(double dens, double x1, double x2, double y1, double y2,
                        double z1, double z2, double xp, double yp, double zp);


/* Calculates the components gravity gradient tensor components caused by a
right rectangular prism using the formulas given in Nagy (2000).

The coordinate system of the input parameters is assumed to be
    x->north, y->east; z->down.

Input values in SI units and returns values in Eotvos!

Parameters:
    * double dens: density of the prism;
    * double x1, x2, y1, ... z2: the borders of the prism;
    * double xp, yp, zp: coordinates of the point P where the effect will be
                         calculated;
*/
extern double prism_gxx(double dens, double x1, double x2, double y1, double y2,
                        double z1, double z2, double xp, double yp, double zp);

extern double prism_gxy(double dens, double x1, double x2, double y1, double y2,
                        double z1, double z2, double xp, double yp, double zp);

extern double prism_gxz(double dens, double x1, double x2, double y1, double y2,
                        double z1, double z2, double xp, double yp, double zp);

extern double prism_gyy(double dens, double x1, double x2, double y1, double y2,
                        double z1, double z2, double xp, double yp, double zp);

extern double prism_gyz(double dens, double x1, double x2, double y1, double y2,
                        double z1, double z2, double xp, double yp, double zp);

extern double prism_gzz(double dens, double x1, double x2, double y1, double y2,
                        double z1, double z2, double xp, double yp, double zp);
/* ************************************************************************** */

#endif
