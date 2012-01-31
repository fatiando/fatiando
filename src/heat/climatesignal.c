/* *****************************************************************************
 Copyright 2012 The Fatiando a Terra Development Team

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

   Calculate residual temperature along a well due to a perturbation in the
   surface temperature. The perturbation can be abrupt or linear.
   
   Author: Leonardo Uieda
   Date: 30 January 2012

   ************************************************************************** */

#include <math.h>

const double FAT_PI = 3.1415926535897932384626433832795;

/* Calculate residual temperature along a well due to an abrupt perturbation in
the surface temperature.

The coordinate system of the input parameters is assumed to be z->down.

Input and output values in SI units.

Parameters:
    * double diffus: thermal diffusivity of the medium;
    * double amp: amplitude of the temperature perturbation;
    * double age: how long ago the perturbation occured;
    * double *zp: array of z coordinates of measuring points along the well;
    * unsigned int n: number of computation points;
    * double *res: vector used to return the calculated effect on the n points
Returns:
    * unsigned int: number of points calculated
*/
unsigned int climatesignal_abrupt(double diffus, double amp, double age,
    double *zp, unsigned int n, double *res)
{
    register unsigned int i;
    
    for(i=0; i < n; i++, res++, zp++)
    {
        *res = amp*(1. - erf((double)(*zp)/sqrt(4.*diffus*age)));
    }
    return i;
}

/* Calculate residual temperature along a well due to an linear perturbation in
the surface temperature.

The coordinate system of the input parameters is assumed to be z->down.

Input and output values in SI units (or in matching units).

Parameters:
    * double diffus: thermal diffusivity of the medium;
    * double amp: amplitude of the temperature perturbation;
    * double age: how long ago the perturbation occured;
    * double *zp: array of z coordinates of measuring points along the well;
    * unsigned int n: number of computation points;
    * double *res: vector used to return the calculated effect on the n points
Returns:
    * unsigned int: number of points calculated
*/
unsigned int climatesignal_linear(double diffus, double amp, double age,
    double *zp, unsigned int n, double *res)
{
    double tmp;
    register unsigned int i;
    
    for(i=0; i < n; i++, res++, zp++)
    {
        tmp = (double)(*zp)/sqrt(4.*diffus*age);
        *res = amp*((1. + 2*tmp*tmp)*erfc(tmp) -
                    2./sqrt(FAT_PI)*tmp*exp(-tmp*tmp));
    }
    return i;
}
