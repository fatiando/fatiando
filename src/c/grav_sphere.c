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

#include <math.h>
#include "grav_sphere.h"


/* Calculates the gz gravity component */
double sphere_gz(double dens, double radius, double xc, double yc, double zc,
                 double xp, double yp, double zp)
{
    /* Variables */
    double mass, dx, dy, dz, r_sqr, res;

    mass = (double)(dens*4.*3.1415926535897931*radius*radius*radius)/3.;

    dx = xc - xp;
    dy = yc - yp;
    dz = zc - zp;

    r_sqr = dx*dx + dy*dy + dz*dz;

    res = (double)(G*SI2MGAL*mass*dz/pow(r_sqr, 1.5));

    return res;
}