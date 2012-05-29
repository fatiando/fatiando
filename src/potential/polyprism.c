/* **************************************************************************

   Calculate the potential fields and derivatives of the 3D prism with polygonal
   crossection. Uses forumla of Plouff (1976)

   Author: Vanderlei Coelho de Olivera Junior

   ************************************************************************** */

#include <math.h>


const double GRAV = 0.00000000006673; /* Gravitational constant in SI */


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
    * double *res: vector used to return the calculated effect on the n points
Returns:
    * int: 0
*/
int polyprism_gz(double dens, double z1, double z2, double *x, double *y,
                 int nvertices, double *xp, double *yp, double *zp, int n,
				 double *res)
{
	register int i, k;
	double gzaux, aux;
	double Xk1, Xk2, Yk1, Yk2, Z1, Z1_quadrado, Z2, Z2_quadrado;
	double auxk1, auxk2, aux1k1, aux1k2, aux2k1, aux2k2;
	double Ak1, Ak2, Bk1, Bk2, Ck1, Ck2, Dk1, Dk2, E1k1, E1k2, E2k1, E2k2;
	double Qk1, Qk2, R1k1, R1k2, R2k1, R2k2, p, p_quadrado;
    double *p2x, *p2y; /* pointers to x and y */

	for (i = 0; i < n; i++)
    {
		*res = 0.0;
        /* 100000 transforms SI to mGal */
        aux = GRAV*dens*100000.0;
        gzaux = 0.0;
        Z1 = z1 - *zp;
        Z2 = z2 - *zp;
        Z1_quadrado = pow(Z1, 2);
        Z2_quadrado = pow(Z2, 2);
        /* Loop over vertices */
        p2x = x;
        p2y = y;
        for (k = 0; k < (nvertices-1); k++)
        {
            Xk1 = *p2x - *xp;
            Xk2 = *(p2x+1) - *xp;
            Yk1 = *p2y - *yp;
            Yk2 = *(p2y+1) - *yp;

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
            #undef Divide_macro
            gzaux += (Z2 - Z1)*(atan(auxk2) - atan(auxk1));
            gzaux += Z2*(atan(aux2k1*auxk1) - atan(aux2k2*auxk2));
            gzaux += Z1*(atan(aux1k2*auxk2) - atan(aux1k1*auxk1));
            #define Divide_macro2(a, b) (((a) + (1E-10))/((b) + (1E-10)))
            gzaux += Dk1*(log(Divide_macro2((E1k1 - Ck1), (E1k1 + Ck1))) -
                          log(Divide_macro2((E2k1 - Ck1), (E2k1 + Ck1))));
            gzaux += Dk2*(log(Divide_macro2((E2k2 - Ck2), (E2k2 + Ck2))) -
                          log(Divide_macro2((E1k2 - Ck2), (E1k2 + Ck2))));
            #undef Divide_macro2
            p2x++;
            p2y++;
        }
        /* Calculate the last edge */
        Xk1 = *p2x - *xp;
        Xk2 = *x - *xp;
        Yk1 = *p2y - *yp;
        Yk2 = *y - *yp;

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
        #undef Divide_macro
        gzaux += (Z2 - Z1)*(atan(auxk2) - atan(auxk1));
        gzaux += Z2*(atan(aux2k1*auxk1) - atan(aux2k2*auxk2));
        gzaux += Z1*(atan(aux1k2*auxk2) - atan(aux1k1*auxk1));
        #define Divide_macro2(a, b) (((a) + (1E-10))/((b) + (1E-10)))
        gzaux += Dk1*(log(Divide_macro2((E1k1 - Ck1), (E1k1 + Ck1))) -
                      log(Divide_macro2((E2k1 - Ck1), (E2k1 + Ck1))));
        gzaux += Dk2*(log(Divide_macro2((E2k2 - Ck2), (E2k2 + Ck2))) -
                      log(Divide_macro2((E1k2 - Ck2), (E1k2 + Ck2))));
        #undef Divide_macro
        gzaux *= aux;
        *res += gzaux;
        res++;
        xp++;
        yp++;
        zp++;
	}
	return 0;
}
