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

   Calculate the gravitational attraction of a 2D body with polygonal vertical
   cross-section using the formula of Talwani et al. (1959)
   
   Talwani, M., J. L. Worzel, and M. Landisman (1959), Rapid Gravity
     Computations for Two-Dimensional Bodies with Application to the
     Mendocino Submarine Fracture Zone, J. Geophys. Res., 64(1), 49-59,
     doi:10.1029/JZ064i001p00049.

   Author: Leonardo Uieda
   Date: 12 January 2012

   ************************************************************************** */

#include <math.h>

/* The gravitational constant (m^3*kg^-1*s^-1) */
#define G 0.00000000006673

/* Conversion factor from SI units to mGal: 1 m/s**2 = 10**5 mGal */
#define SI2MGAL 100000.0


/* Calculate the gravitational attraction of a 2D body with polygonal vertical
   cross-section using the formula of Talwani et al. (1959)

The coordinate system of the input parameters is assumed to be z->down.

Input values in SI units and returns values in mGal.

REMEMBER: Vertices must be clockwise or the gz sign will be inverted!

Parameters:
    * double dens: density of the prism;
    * double *x, *z: x and z coordinates of the vertices;
    * unsigned int m: number of vertices of the polygonal cross-section;
    * double *xp, *zp: coordinates of the computation points;
    * unsigned int n: number of computation points;
    * double *res: vector used to return the calculated effect on the n points
Returns:
    * unsigned int: number of points calculated
*/
unsigned int talwani_gz(double dens, double *x, double *z, unsigned int m,
                        double *xp, double *zp, unsigned int n, double *res)
{
    double *px, *pz;
    double xv, zv, xvp1, zvp1, theta_v, theta_vp1, phi_v, ai, tmp;
    int flag;
    register unsigned int i, v;
    
    for(i=0; i < n; i++, res++, xp++, zp++)
    {
        flag = 0;
        *res = 0;
        tmp = 0;
        xvp1 = *x - *xp;
        zvp1 = *z - *zp;  
        px = x;
        pz = z;  
        for(v=0; v < m; v++)
        {
            xv = xvp1;
            zv = zvp1;
            /* The last vertice pairs with the first one */
            if(v == m - 1)
            {
                xvp1 = *x - *xp;
                zvp1 = *z - *zp;
            }
            else
            {
                xvp1 = *(++px) - *xp;
                zvp1 = *(++pz) - *zp;                
            }
            theta_v = atan2(zv, xv); 
            theta_vp1 = atan2(zvp1, xvp1); 
            phi_v = atan2(zvp1 - zv, xvp1 - xv); 
            ai = xvp1 + (zvp1)*((double)(xvp1 - xv)/(zv - zvp1));            
            if(theta_v < 0)
            {
                theta_v += 3.1415926535897932384626433832795;
            }
            if(theta_vp1 < 0)
            {
                theta_vp1 += 3.1415926535897932384626433832795;
            }            
            if(xv == 0)
            {
                /* 1.570796327 = pi/2 */ 
                tmp = -ai*sin(phi_v)*cos(phi_v)*(theta_vp1 -
                    1.57079632679489661923 + tan(phi_v)*log(
                        cos(theta_vp1)*(tan(theta_vp1)- tan(phi_v))));
                flag = 1;
            }             
            if(xvp1 == 0)
            {
                tmp = ai*sin(phi_v)*cos(phi_v)*(theta_v -
                    1.57079632679489661923 + tan(phi_v)*log(
                        cos(theta_v)*(tan(theta_v) - tan(phi_v))));         
                flag = 2;
            }            
            if(zv == zvp1)
            {
                tmp = zv*(theta_vp1 - theta_v);
                flag = 3;
            }            
            if(xv == xvp1)
            {
                tmp = xv*(log((double)cos(theta_v)/cos(theta_vp1)));
                flag = 4;
            }            
            if((theta_v == theta_vp1) || (xv == 0. && zv == 0.) ||
               (xvp1 == 0. && zvp1 == 0.))
            {
                tmp = 0;
                flag = 5;
            }            
            if(!flag)
            { 
                tmp = ai*sin(phi_v)*cos(phi_v)*(theta_v - theta_vp1 +
                    tan(phi_v)*log((double)
                        (cos(theta_v)*(tan(theta_v) - tan(phi_v)))/
                        (cos(theta_vp1)*(tan(theta_vp1) - tan(phi_v)))));
            }
            *res += tmp;
        }
        *res *= SI2MGAL*2.0*G*dens;
    }
    return i;
}
