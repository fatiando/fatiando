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

   Set of functions that calculate:
   *  the gravitational potential and its first and second derivatives for the
      right rectangular prism using the formulas in Nagy et al. (2000)

   Author: Leonardo Uieda
   Date: 01 March 2010

   ************************************************************************** */

#include <math.h>

/* The gravitational constant (m^3*kg^-1*s^-1) */
#define G 0.00000000006673

/* Conversion factor from SI units to Eotvos: 1 /s**2 = 10**9 Eotvos */
#define SI2EOTVOS 1000000000.0

/* Conversion factor from SI units to mGal: 1 m/s**2 = 10**5 mGal */
#define SI2MGAL 100000.0


/* The following functions calculate the gravitational potential and its first
and second derivatives caused by a right rectangular prism using the formulas
given in Nagy et al. (2000).

The coordinate system of the input parameters is assumed to be
    x->north, y->east; z->down.

Input values in SI units and returns values in:
* gx, gy, gz = mGal
* gxx, gxy, gxz, gyy, etc. = Eotvos

Parameters:
    * double dens: density of the prism;
    * double x1, x2, y1, ... z2: the borders of the prism;
    * double *xp, *yp, *zp: coordinates of the computation points
    * unsigned int n: number of computation points
    * double *res: vector used to return the calculated effect on the n points
Returns:
    * unsigned int: number of points calculated
*/


int prism_gz(double dens, double x1, double x2, double y1, double y2,
             double z1, double z2, double *xp, double *yp, double *zp,
             unsigned int n, double *res)
{
    double r,
           deltax1, deltax2, deltay1, deltay2, deltaz1, deltaz2;
    register unsigned int i;

    for(i=0; i < n; i++)
    {
        /* First thing to do is make P the origin of the coordinate system */
        deltax1 = x1 - *xp;
        deltax2 = x2 - *xp;
        deltay1 = y1 - *yp;
        deltay2 = y2 - *yp;
        deltaz1 = z1 - *zp;
        deltaz2 = z2 - *zp;
        *res = 0;
        /* Evaluate the integration limits */
        r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1);
        *res += 1*(deltax1*log(deltay1 + r) + deltay1*log(deltax1 + r) -
                deltaz1*atan2(deltax1*deltay1, deltaz1*r));
        r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1);
        *res += -1*(deltax2*log(deltay1 + r) + deltay1*log(deltax2 + r) -
                deltaz1*atan2(deltax2*deltay1, deltaz1*r));
        r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1);
        *res += -1*(deltax1*log(deltay2 + r) + deltay2*log(deltax1 + r) -
                deltaz1*atan2(deltax1*deltay2, deltaz1*r));
        r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1);
        *res += 1*(deltax2*log(deltay2 + r) + deltay2*log(deltax2 + r) -
                deltaz1*atan2(deltax2*deltay2, deltaz1*r));
        r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2);
        *res += -1*(deltax1*log(deltay1 + r) + deltay1*log(deltax1 + r) -
                deltaz2*atan2(deltax1*deltay1, deltaz2*r));
        r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2);
        *res += 1*(deltax2*log(deltay1 + r) + deltay1*log(deltax2 + r) -
                deltaz2*atan2(deltax2*deltay1, deltaz2*r));
        r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2);
        *res += 1*(deltax1*log(deltay2 + r) + deltay2*log(deltax1 + r) -
                deltaz2*atan2(deltax1*deltay2, deltaz2*r));
        r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2);
        *res += -1*(deltax2*log(deltay2 + r) + deltay2*log(deltax2 + r) -
                deltaz2*atan2(deltax2*deltay2, deltaz2*r));
        *res *= G*SI2MGAL*dens;
        res++;
        xp++;
        yp++;
        zp++;
    }
    return i;
}

int prism_gxx(double dens, double x1, double x2, double y1, double y2,
              double z1, double z2, double *xp, double *yp, double *zp,
              unsigned int n, double *res)
{
    double r,
           deltax1, deltax2, deltay1, deltay2, deltaz1, deltaz2;
    register unsigned int i;

    for(i=0; i < n; i++)
    {
        /* First thing to do is make P the origin of the coordinate system */
        deltax1 = x1 - *xp;
        deltax2 = x2 - *xp;
        deltay1 = y1 - *yp;
        deltay2 = y2 - *yp;
        deltaz1 = z1 - *zp;
        deltaz2 = z2 - *zp;
        *res = 0;
        /* Evaluate the integration limits */
        r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1);
        *res += 1*atan2(deltay1*deltaz1, deltax1*r);
        r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1);
        *res += -1*atan2(deltay1*deltaz1, deltax2*r);
        r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1);
        *res += -1*atan2(deltay2*deltaz1, deltax1*r);
        r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1);
        *res += 1*atan2(deltay2*deltaz1, deltax2*r);
        r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2);
        *res += -1*atan2(deltay1*deltaz2, deltax1*r);
        r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2);
        *res += 1*atan2(deltay1*deltaz2, deltax2*r);
        r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2);
        *res += 1*atan2(deltay2*deltaz2, deltax1*r);
        r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2);
        *res += -1*atan2(deltay2*deltaz2, deltax2*r);
        *res *= G*SI2EOTVOS*dens;
        res++;
        xp++;
        yp++;
        zp++;
    }
    return i;
}

int prism_gxy(double dens, double x1, double x2, double y1, double y2,
              double z1, double z2, double *xp, double *yp, double *zp,
              unsigned int n, double *res)
{
    double r,
           deltax1, deltax2, deltay1, deltay2, deltaz1, deltaz2;
    register unsigned int i;

    for(i=0; i < n; i++)
    {
        /* First thing to do is make P the origin of the coordinate system */
        deltax1 = x1 - *xp;
        deltax2 = x2 - *xp;
        deltay1 = y1 - *yp;
        deltay2 = y2 - *yp;
        deltaz1 = z1 - *zp;
        deltaz2 = z2 - *zp;
        *res = 0;
        /* Evaluate the integration limits */
        r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1);
        *res += 1*(-1*log(deltaz1 + r));
        r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1);
        *res += -1*(-1*log(deltaz1 + r));
        r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1);
        *res += -1*(-1*log(deltaz1 + r));
        r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1);
        *res += 1*(-1*log(deltaz1 + r));
        r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2);
        *res += -1*(-1*log(deltaz2 + r));
        r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2);
        *res += 1*(-1*log(deltaz2 + r));
        r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2);
        *res += 1*(-1*log(deltaz2 + r));
        r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2);
        *res += -1*(-1*log(deltaz2 + r));
        *res *= G*SI2EOTVOS*dens;
        res++;
        xp++;
        yp++;
        zp++;
    }
    return i;
}

int prism_gxz(double dens, double x1, double x2, double y1, double y2,
              double z1, double z2, double *xp, double *yp, double *zp,
              unsigned int n, double *res)
{
    double r,
           deltax1, deltax2, deltay1, deltay2, deltaz1, deltaz2;
    register unsigned int i;

    for(i=0; i < n; i++)
    {
        /* First thing to do is make P the origin of the coordinate system */
        deltax1 = x1 - *xp;
        deltax2 = x2 - *xp;
        deltay1 = y1 - *yp;
        deltay2 = y2 - *yp;
        deltaz1 = z1 - *zp;
        deltaz2 = z2 - *zp;
        *res = 0;
        /* Evaluate the integration limits */
        r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1);
        *res += 1*(-1*log(deltay1 + r));
        r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1);
        *res += -1*(-1*log(deltay1 + r));
        r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1);
        *res += -1*(-1*log(deltay2 + r));
        r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1);
        *res += 1*(-1*log(deltay2 + r));
        r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2);
        *res += -1*(-1*log(deltay1 + r));
        r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2);
        *res += 1*(-1*log(deltay1 + r));
        r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2);
        *res += 1*(-1*log(deltay2 + r));
        r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2);
        *res += -1*(-1*log(deltay2 + r));
        *res *= G*SI2EOTVOS*dens;
        res++;
        xp++;
        yp++;
        zp++;
    }
    return i;
}

int prism_gyy(double dens, double x1, double x2, double y1, double y2,
              double z1, double z2, double *xp, double *yp, double *zp,
              unsigned int n, double *res)
{
    double r,
           deltax1, deltax2, deltay1, deltay2, deltaz1, deltaz2;
    register unsigned int i;

    for(i=0; i < n; i++)
    {
        /* First thing to do is make P the origin of the coordinate system */
        deltax1 = x1 - *xp;
        deltax2 = x2 - *xp;
        deltay1 = y1 - *yp;
        deltay2 = y2 - *yp;
        deltaz1 = z1 - *zp;
        deltaz2 = z2 - *zp;
        *res = 0;
        /* Evaluate the integration limits */
        r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1);
        *res += 1*(atan2(deltaz1*deltax1, deltay1*r));
        r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1);
        *res += -1*(atan2(deltaz1*deltax2, deltay1*r));
        r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1);
        *res += -1*(atan2(deltaz1*deltax1, deltay2*r));
        r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1);
        *res += 1*(atan2(deltaz1*deltax2, deltay2*r));
        r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2);
        *res += -1*(atan2(deltaz2*deltax1, deltay1*r));
        r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2);
        *res += 1*(atan2(deltaz2*deltax2, deltay1*r));
        r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2);
        *res += 1*(atan2(deltaz2*deltax1, deltay2*r));
        r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2);
        *res += -1*(atan2(deltaz2*deltax2, deltay2*r));
        *res *= G*SI2EOTVOS*dens;
        res++;
        xp++;
        yp++;
        zp++;
    }
    return i;
}

int prism_gyz(double dens, double x1, double x2, double y1, double y2,
              double z1, double z2, double *xp, double *yp, double *zp,
              unsigned int n, double *res)
{
    double r,
           deltax1, deltax2, deltay1, deltay2, deltaz1, deltaz2;
    register unsigned int i;

    for(i=0; i < n; i++)
    {
        /* First thing to do is make P the origin of the coordinate system */
        deltax1 = x1 - *xp;
        deltax2 = x2 - *xp;
        deltay1 = y1 - *yp;
        deltay2 = y2 - *yp;
        deltaz1 = z1 - *zp;
        deltaz2 = z2 - *zp;
        *res = 0;
        /* Evaluate the integration limits */
        r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1);
        *res += 1*(-1*log(deltax1 + r));
        r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1);
        *res += -1*(-1*log(deltax2 + r));
        r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1);
        *res += -1*(-1*log(deltax1 + r));
        r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1);
        *res += 1*(-1*log(deltax2 + r));
        r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2);
        *res += -1*(-1*log(deltax1 + r));
        r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2);
        *res += 1*(-1*log(deltax2 + r));
        r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2);
        *res += 1*(-1*log(deltax1 + r));
        r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2);
        *res += -1*(-1*log(deltax2 + r));
        *res *= G*SI2EOTVOS*dens;
        res++;
        xp++;
        yp++;
        zp++;
    }
    return i;
}

int prism_gzz(double dens, double x1, double x2, double y1, double y2,
              double z1, double z2, double *xp, double *yp, double *zp,
              unsigned int n, double *res)
{
    double r,
           deltax1, deltax2, deltay1, deltay2, deltaz1, deltaz2;
    register unsigned int i;

    for(i=0; i < n; i++)
    {
        /* First thing to do is make P the origin of the coordinate system */
        deltax1 = x1 - *xp;
        deltax2 = x2 - *xp;
        deltay1 = y1 - *yp;
        deltay2 = y2 - *yp;
        deltaz1 = z1 - *zp;
        deltaz2 = z2 - *zp;
        *res = 0;
        /* Evaluate the integration limits */
        r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1);
        *res += 1*(atan2(deltax1*deltay1, deltaz1*r));
        r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1);
        *res += -1*(atan2(deltax2*deltay1, deltaz1*r));
        r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1);
        *res += -1*(atan2(deltax1*deltay2, deltaz1*r));
        r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1);
        *res += 1*(atan2(deltax2*deltay2, deltaz1*r));
        r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2);
        *res += -1*(atan2(deltax1*deltay1, deltaz2*r));
        r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2);
        *res += 1*(atan2(deltax2*deltay1, deltaz2*r));
        r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2);
        *res += 1*(atan2(deltax1*deltay2, deltaz2*r));
        r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2);
        *res += -1*(atan2(deltax2*deltay2, deltaz2*r));
        *res *= G*SI2EOTVOS*dens;
        res++;
        xp++;
        yp++;
        zp++;
    }
    return i;
}
