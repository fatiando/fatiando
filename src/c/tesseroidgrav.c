/* **************************************************************************

   This module contains a set of functions that calculate the gravitational
   potential and its first and second derivatives for a tesseroid. The integrals
   are solved numerically with a Gauss-Legendre Quadrature (GLQ). The numerical
   integration can be performed for the whole volume integral or only for the
   surface integral. The later is faster but less stable when calculating
   directly above the tesseroid.

   Author: Leonardo Uieda
   Date: 07 May 2010

   ************************************************************************** */

#include <math.h>
#include <stdlib.h>
#include "tesseroidgrav.h"


/* TESS_GXX2D */
/* ************************************************************************** */
/* Calculates the Gxx component of the gravity gradient tensor caused by
a tesseroid using GLQ for numerical integration of the SURFACE integral.

The coordinate system of the input parameters is assumed to be
    x->north, y->east; z->down.

Input values in SI units and returns values in Eotvos!

Parameters:
    * dens: density of the tesseroid (kg m^-3)
    * lon1, lon2, lat1, lat2: longitude and latitude borders of the tesseroid
                              (degrees);
    * z1, z2: top and bottom, respectively, of the tesseroid (m)
    *         Note: Remember that z points down!
    * lonp, latp: longitude and latitude of the computation point (degrees);
    * zp: height of the computation point (m)
          Note: Remember that z points down! So a height of 100m is zp=-100
    * glq_order: order of the GLQ used to integrate the surface integral
    * nodes: array with the GLQ nodes
    * weights: array with the GLQ weights
*/
double tess_gxx2d(double dens, double lon1, double lon2, double lat1,
       double lat2, double z1, double z2, double lonp, double latp,
       double zp, int glq_order, double *nodes, double *weights)
{
    double *pnlon, *pnlat, *pwlon, *pwlat;
    double lambl, phil, r1, r2, r, lamb, phi;
    double result, kernel;
    double alon, blon, alat, blat;
    register int i, j;
    /* Auxiliary variables for the calculations */
    double r_2, r1_2, r2_2,
           cosPhil, cosPhi, sinPhil, sinPhi,
           cosLambLambl, cosPsi, cosPsiPhi, cosPsiPhi_2,
           l1, l2, l1_2, l2_2, r1l1, r2l2,
           cosPsi_2_1, lntop, lnbot, ln,
           t1, t2, t3, t4, t5, t6, t7,
           Kphi_2, Kr;

    lon1 = DEG2RAD*lon1;
    lon2 = DEG2RAD*lon2;
    lat1 = DEG2RAD*lat1;
    lat2 = DEG2RAD*lat2;
    lamb = DEG2RAD*lonp;
    phi = DEG2RAD*latp;
    r1 = MEAN_EARTH_RADIUS - z2;
    r2 = MEAN_EARTH_RADIUS - z1;
    r = MEAN_EARTH_RADIUS - zp;

    /* Angular and linear scale factors for the longitude
     * and latitude nodes */
    blon = 0.5*(lon2 + lon1);
    alon = 0.5*(lon2 - lon1);
    blat = 0.5*(lat2 + lat1);
    alat = 0.5*(lat2 - lat1);

    pnlon = nodes;
    pwlon = weights;
    for(i=0; i < glq_order; i++)
    {

        lambl = alon*(*pnlon) + blon;

        pnlat = nodes;
        pwlat = weights;
        for(j=0; j < glq_order; j++)
        {
            phil = alat*(*pnlat) + blat;

            r_2 = r*r;
            r1_2 = r1*r1;
            r2_2 = r2*r2;
            cosPhil = cos(phil);
            cosPhi = cos(phi);
            sinPhil = sin(phil);
            sinPhi = sin(phi);
            cosLambLambl = cos(lamb - lambl);
            cosPsi = sinPhi*sinPhil + cosPhi*cosPhil*cosLambLambl;
            cosPsiPhi = cosPhi*sinPhil - sinPhi*cosPhil*cosLambLambl;
            cosPsiPhi_2 = cosPsiPhi*cosPsiPhi;
            l1 = sqrt( r_2 + r1_2 - 2*r*r1*cosPsi );
            l2 = sqrt( r_2 + r2_2 - 2*r*r2*cosPsi );
            l1_2 = l1*l1;
            l2_2 = l2*l2;
            r1l1 = (double)r1/l1;
            r2l2 = (double)r2/l2;
            lntop = l2 + r2 - (r*cosPsi);
            lnbot = l1 + r1 - (r*cosPsi);
            cosPsi_2_1 = (3*cosPsi*cosPsi) - 1;
            ln = log(fabs((double)lntop / lnbot));

            /* Kphi**2 */
            t1 = (r2l2*r2l2*r/l2)*(r*r2*cosPsiPhi_2 - l2_2*cosPsi);
            t2 = (r1l1*r1l1*r/l1)*(r*r1*cosPsiPhi_2 - l1_2*cosPsi);
            t3 = (double)3*r_2*cosPsiPhi_2*( 2*(r1l1 - r2l2) +
                      r*cosPsi*(r1l1*r1l1/l1 - r2l2*r2l2/l2) );
            t4 = 3*r*cosPsi*(l2 - l1 + cosPsi*(r*r1l1 - r*r2l2));
            t5 = 6*r_2*ln*(cosPsiPhi_2 - cosPsi*cosPsi);
            t6 = (double)r_2*r*cosPsi*(12*cosPsiPhi_2 - cosPsi_2_1)*(
                    (r1 + l1)/(l1*lnbot) - (r2 + l2)/(l2*lntop) );
            t7 = (double)r_2*r_2*cosPsiPhi_2*cosPsi_2_1*(
                    (r1*lnbot - (r1 + l1)*(r1l1*lnbot + r1 + l1)) /
                    (l1*l1*lnbot*lnbot) -
                    (r2*lntop - (r2 + l2)*(r2l2*lntop + r2 + l2)) /
                    (l2*l2*lntop*lntop) );
            Kphi_2 = 0.5*( t2 - t1 + t3 - t4 + t5 + t6 - t7 );

            /* Kr */
            Kr = (double)(r2*l2 - r1*l1 + 3*r*cosPsi*(l2 - l1) +
                  r_2*cosPsi_2_1*ln - r2l2*r2_2 + r1l1*r1_2) / r;

            kernel = (double)cosPhil*(Kphi_2 + r*Kr) / r_2;

            result = (*pwlon)*(*pwlat)*kernel;

            pnlat++;
            pwlat++;
        }

        pnlon++;
        pwlon++;
    }

    return SI2EOTVOS*G*dens*0.25*(lon2 - lon1)*(lat2 - lat1)*result;
}
