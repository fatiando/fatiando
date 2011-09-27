/* *****************************************************************************
 Copyright 2011 The Fatiando a Terra Development Team

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

   Potential field transformations.

   Author: Leonardo Uieda
   Date: 27 Sep 2011

   ************************************************************************** */

#include <math.h>

const double PI = 3.1415926535897932384626433832795;

/* Perform the upward continuation of gravity data using the analytical formula

Parameters:
    * double dens: density of the prism;
    * double x1, x2, y1, ... z2: the borders of the prism;
    * double *xp, *yp, *zp: coordinates of the computation points
    * unsigned int n: number of computation points
    * double *res: vector used to return the calculated effect on the n points
Returns:
    * unsigned int: number of points calculated
*/


int upcontinue(double *xp, double *yp, double *zp, double *newz, double *gz,
               unsigned int n, double dx, double dy, double *gzcont)
{
    register unsigned int i, j;
    double *x, *y, *z, *xl, *yl, *zl, *g, area, oneover_l;

    area = dx*dy;
    x = xp;
    y = yp;
    z = zp;
    for(i=0; i < n; i++)
    {
        xl = xp;
        yl = yp;
        zl = newz;
        *gzcont = 0;
        g = gz;
        for(j=0; j < n; j++)
        {
            oneover_l = pow((*x-*xl)*(*x-*xl) + (*y-*yl)*(*y-*yl) +
                            (*z-*zl)*(*z-*zl), -1.5);
            *gzcont += (*g)*oneover_l*area;
            xl++;
            yl++;
            zl++;
            g++;
        }
        #define FAT_ABS(x) ((x) < 0 ? -1*(x) : (x))
        *gzcont *= FAT_ABS(*zp - *newz)/(2*PI);
        #undef FAT_ABS
        gzcont++;
        x++;
        y++;
        z++;
    }
    return i;
}
