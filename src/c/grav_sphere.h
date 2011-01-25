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

   Functions that calculate the gravitational potential and its first and second
   derivatives for a sphere.

   Author: Leonardo Uieda
   Date: 29 Nov 2010

   ************************************************************************** */

#ifndef _GRAV_SPHERE_H_
#define _GRAV_SPHERE_H_

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


/* Calculates the components of the gravity attraction caused by a sphere.

The coordinate system of the input parameters is assumed to be
    x->north, y->east; z->down.

Input values in SI units and returns values in mGal!

Parameters:
    * double dens: density of the sphere;
    * double radius: of the sphere;
    * double xc, yc, zc: coordinates of the center of the sphere;
    * double xp, yp, zp: coordinates of the point P where the effect will be
                         calculated;
*/
extern double sphere_gz(double dens, double radius, double xc, double yc,
                        double zc, double xp, double yp, double zp);


/* Calculates the components of the gravity gradient tensor caused by a sphere.

The coordinate system of the input parameters is assumed to be
    x->north, y->east; z->down.

Input values in SI units and returns values in Eotvos!

Parameters:
    * double dens: density of the sphere;
    * double radius: of the sphere;
    * double xc, yc, zc: coordinates of the center of the sphere;
    * double xp, yp, zp: coordinates of the point P where the effect will be
                         calculated;
*/
extern double sphere_gxx(double dens, double radius, double xc, double yc,
                         double zc, double xp, double yp, double zp);

extern double sphere_gyy(double dens, double radius, double xc, double yc,
                         double zc, double xp, double yp, double zp);

extern double sphere_gzz(double dens, double radius, double xc, double yc,
                         double zc, double xp, double yp, double zp);
/* ************************************************************************** */

#endif
