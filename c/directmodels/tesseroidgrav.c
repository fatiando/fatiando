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
#include "glq.h"


/* TESS_POT2D */
/* ************************************************************************** */
/* Calculates the gravitational potential caused by a tesseroid using GLQ for
numerical integration of the SURFACE integral.

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
*/
double tess_pot2d(double dens, double lon1, double lon2, double lat1,
       double lat2, double z1, double z2, double lonp, double latp,
       double zp, int glq_order)
{
    double *nodes, *weights, *lon_nodes, *lat_nodes, plon, plat, pweight;
    double result;
    int i, j;

    /* Allocate memory for the nodes, weights and scaled nodes */
    nodes = (double *)malloc(glq_order*sizeof(double));
    if(!nodes)
    {
        return 0;
    }

    weights = (double *)malloc(glq_order*sizeof(double));
    if(!weights)
    {
        free(nodes);
        return 0;
    }

    lon_nodes = (double *)malloc(glq_order*sizeof(double));
    if(!lon_nodes)
    {
        free(nodes);
        free(weights);
        return 0;
    }

    lat_nodes = (double *)malloc(glq_order*sizeof(double));
    if(!lat_nodes)
    {
        free(nodes);
        free(weights);
        free(lon_nodes);
        return 0;
    }

    /* Calculate the nodes and weights and scale them */
    glq_nodes(order, nodes);
    glq_weights(order, nodes, weights);
    glq_scale(DEG2RAD*lon1, DEG2RAD*lon2, order, nodes, lon_nodes);
    glq_scale(DEG2RAD*lat1, DEG2RAD*lat2, order, nodes, lat_nodes);

    /* The GLQ summation */
    for(i=0, plon=lon_nodes; i < order; i++, plon++)
    {
        for(j=0, plat=lat_nodes; j < order; j++, plat++)
        {

        }
    }


}
