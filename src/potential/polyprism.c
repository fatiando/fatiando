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

   Calculate the potential fields and derivatives of the 3D prism with polygonal
   crossection. Uses forumla of Plouff (1976)

   Author: Vanderlei Coelho de Olivera Junior
   Date: 28 Sep 2011

   ************************************************************************** */

#include <math.h>


const double GRAV = 0.0667; /* Gravitational constant */


/* The following functions calculate the gravitational potential and its first
and second derivatives caused by a prism with polygonal crossection
using the formula given in Plouff (1976).

The coordinate system of the input parameters is assumed to be
    x->north, y->east; z->down.

Input values in SI units and returns values in:
* gx, gy, gz = mGal
* gxx, gxy, gxz, gyy, etc. = Eotvos

Parameters:
    * double dens: density of the prism
    * double z1, z2: top and bottom of the prism
    * double *x, *y: x and y coordinates of the vertices of the prism
    * int nvertices: number of vertices
    * double *xp, *xp, *zp: coordinates of the computation points
    * int n: number of computation points
    * double *g: vector used to return the calculated effect on the N points
Returns:
    * int: 0
*/

int polyprism_gz(double dens, double z1, double z2, double *x, double *y,
                 int nvertices, double *xp, double *yp, int n, double *g)
{
	register int i, k;
	double gaux, aux, auxk1, auxk2, aux1k1, aux1k2, aux2k1, aux2k2;
	double Ak1, Ak2, Bk1, Bk2, Ck1, Ck2, Dk1, Dk2, E1k1, E1k2, E2k1, E2k2;
	double Xk1, Xk2, Yk1, Yk2, Z1, Z1_quadrado, Z2, Z2_quadrado, Qk1, Qk2;
	double R1k1, R1k2, R2k1, R2k2, p, p_quadrado;

    /* Loop over points */
	for (i = 0; i < n; i++)
    {
		g[i] = 0.0;
        /* 0.001 is to convert from kg/m3 to g/cm3 */
        aux = GRAV*(dens*0.001)*100;
        gaux = 0.0;
        Z1 = z1*0.001;
        Z2 = z2*0.001;
        Z1_quadrado = pow(Z1, 2);
        Z2_quadrado = pow(Z2, 2);
        /* Loop over vertices */
        for (k = 0; k < (nvertices - 1); k++)
        {
            Xk1 = (x[k] - xp[i])*0.001;
            Xk2 = (x[k+1] - xp[i])*0.001;
            Yk1 = (y[k] - yp[i])*0.001;
            Yk2 = (y[k+1] - yp[i])*0.001;

            p = (Xk1*Yk2) - (Xk2*Yk1);
            p_quadrado = pow(p, 2);
            Qk1 = ((Yk2 - Yk1)*Yk1) + ((Xk2 - Xk1)*Xk1);
            Qk2 = ((Yk2 - Yk1)*Yk2) + ((Xk2 - Xk1)*Xk2);

            Ak1 = pow(Xk1, 2) + pow(Yk1, 2);
            Ak2 = pow(Xk2, 2) + pow(Yk2, 2);

            R1k1 = Ak1 + Z1_quadrado;
            R1k1 = pow(R1k1, 0.5);
            R1k2 = Ak2 + Z1_quadrado;
            R1k2 = pow(R1k2, 0.5);
            R2k1 = Ak1 + Z2_quadrado;
            R2k1 = pow(R2k1, 0.5);
            R2k2 = Ak2 + Z2_quadrado;
            R2k2 = pow(R2k2, 0.5);

            Ak1 = pow(Ak1, 0.5);
            Ak2 = pow(Ak2, 0.5);

            Bk1 = pow(Qk1, 2) + p_quadrado;
            Bk1 = pow(Bk1, 0.5);
            Bk2 = pow(Qk2, 2) + p_quadrado;
            Bk2 = pow(Bk2, 0.5);

            Ck1 = Qk1*Ak1;
            Ck2 = Qk2*Ak2;

            #define Divide_macro(a, b) ((a)/((b) + (1E-10)))
            Dk1 = Divide_macro(p, 2.0);
            Dk2 = Dk1;
            Dk1 *= Divide_macro(Ak1, Bk1);
            Dk2 *= Divide_macro(Ak2, Bk2);

            E1k1 = R1k1*Bk1;
            E1k2 = R1k2*Bk2;
            E2k1 = R2k1*Bk1;
            E2k2 = R2k2*Bk2;

            auxk1 = Divide_macro(Qk1, p);
            auxk2 = Divide_macro(Qk2, p);
            aux1k1 = Divide_macro(Z1, R1k1);
            aux1k2 = Divide_macro(Z1, R1k2);
            aux2k1 = Divide_macro(Z2, R2k1);
            aux2k2 = Divide_macro(Z2, R2k2);
            #udef Divide_macro

            gaux += (Z2 - Z1)*(atan(auxk2) - atan(auxk1));
            gaux += Z2*(atan(aux2k1*auxk1) - atan(aux2k2*auxk2));
            gaux += Z1*(atan(aux1k2*auxk2) - atan(aux1k1*auxk1));

            #define Divide_macro2(a, b) (((a) + (1E-10))/((b) + (1E-10)))
            gaux += Dk1*(log(Divide_macro2((E1k1 - Ck1), (E1k1 + Ck1))) -
                         log(Divide_macro2((E2k1 - Ck1), (E2k1 + Ck1))));
            gaux += Dk2*(log(Divide_macro2((E2k2 - Ck2), (E2k2 + Ck2))) -
                         log(Divide_macro2((E1k2 - Ck2), (E1k2 + Ck2))));
            #udef Divide_macro2
        }
        /* The last edge */
        Xk1 = (x[nvertices-1] - xp[i])*0.001;
        Xk2 = (x[0] - xp[i])*0.001;
        Yk1 = (y[nvertices-1] - yp[i])*0.001;
        Yk2 = (y[0] - yp[i])*0.001;

        p = (Xk1*Yk2) - (Xk2*Yk1);
        p_quadrado = pow(p, 2);
        Qk1 = ((Yk2 - Yk1)*Yk1) + ((Xk2 - Xk1)*Xk1);
        Qk2 = ((Yk2 - Yk1)*Yk2) + ((Xk2 - Xk1)*Xk2);

        Ak1 = pow(Xk1, 2) + pow(Yk1, 2);
        Ak2 = pow(Xk2, 2) + pow(Yk2, 2);

        R1k1 = Ak1 + Z1_quadrado;
        R1k1 = pow(R1k1, 0.5);
        R1k2 = Ak2 + Z1_quadrado;
        R1k2 = pow(R1k2, 0.5);
        R2k1 = Ak1 + Z2_quadrado;
        R2k1 = pow(R2k1, 0.5);
        R2k2 = Ak2 + Z2_quadrado;
        R2k2 = pow(R2k2, 0.5);

        Ak1 = pow(Ak1, 0.5);
        Ak2 = pow(Ak2, 0.5);

        Bk1 = pow(Qk1, 2) + p_quadrado;
        Bk1 = pow(Bk1, 0.5);
        Bk2 = pow(Qk2, 2) + p_quadrado;
        Bk2 = pow(Bk2, 0.5);

        Ck1 = Qk1*Ak1;
        Ck2 = Qk2*Ak2;

        #define Divide_macro(a, b) ((a)/((b) + (1E-10)))
        Dk1 = Divide_macro(p, 2.0);
        Dk2 = Dk1;
        Dk1 *= Divide_macro(Ak1, Bk1);
        Dk2 *= Divide_macro(Ak2, Bk2);

        E1k1 = R1k1*Bk1;
        E1k2 = R1k2*Bk2;
        E2k1 = R2k1*Bk1;
        E2k2 = R2k2*Bk2;

        auxk1 = Divide_macro(Qk1, p);
        auxk2 = Divide_macro(Qk2, p);
        aux1k1 = Divide_macro(Z1, R1k1);
        aux1k2 = Divide_macro(Z1, R1k2);
        aux2k1 = Divide_macro(Z2, R2k1);
        aux2k2 = Divide_macro(Z2, R2k2);
        #udef Divide_macro

        gaux += (Z2 - Z1)*(atan(auxk2) - atan(auxk1));
        gaux += Z2*(atan(aux2k1*auxk1) - atan(aux2k2*auxk2));
        gaux += Z1*(atan(aux1k2*auxk2) - atan(aux1k1*auxk1));

        #define Divide_macro2(a, b) (((a) + (1E-10))/((b) + (1E-10)))
        gaux += Dk1*(log(Divide_macro2((E1k1 - Ck1), (E1k1 + Ck1))) -
                     log(Divide_macro2((E2k1 - Ck1), (E2k1 + Ck1))));
        gaux += Dk2*(log(Divide_macro2((E2k2 - Ck2), (E2k2 + Ck2))) -
                     log(Divide_macro2((E1k2 - Ck2), (E1k2 + Ck2))));
        #udef Divide_macro2

        gaux *= aux;
        g[i] += gaux;
	}
	return 0;
}
