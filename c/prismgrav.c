/* **************************************************************************

   This module contains a set of functions that calculate the gravitational
   potential and its first and second derivatives for the rectangular prism
   using the formulas in Nagy (2000).

   Author: Leonardo Uieda
   Date: 01 March 2010
   Last Update: $DATE: $
   $REVISION: $

   ************************************************************************** */

#include <math.h>
#include "prismgrav.h"

/* Calculates the gxx gravity gradient tensor component cause by a prism. */
double prism_gxx(double dens, double x1, double x2, double y1, double y2,
                 double z1, double z2, double xp, double yp, double zp)
{
    /* Variables */
    double r,
           res = 0,
           /* These are used for a dirty little trick to evaluate the primitive
              over the integration limits */
           x[2], y[2], z[2];
    int i, j, k;

    /* First thing to do is make P the origin of the coordinate system */
    x[0] = x1 - xp;
    x[1] = x2 - xp;
    y[0] = y1 - yp;
    y[1] = y2 - yp;
    z[0] = z1 - zp;
    z[1] = z2 - zp;

    /* Do the little summation to evaluate the primitive over the integration
       limits */
    for(k=0; k<=1; k++)
    {
        for(j=0; j<=1; j++)
        {
            for(i=0; i<=1; i++)
            {
                r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);

                /* There should be a minus sign in front of pow but it cancels
                   with the one that should be in front of atan */
                res += pow(-1, i + j + k)*atan2(y[j]*z[k], x[i]*r);
            }
        }
    }

    /* Now all that is left is to multiply res by the gravitational constant and
       density and convert it to Eotvos units */
    res *= G*SI2EOTVOS*dens;

    return res;
}


/* Calculates the gxy gravity gradient tensor component cause by a prism. */
double prism_gxy(double dens, double x1, double x2, double y1, double y2,
                 double z1, double z2, double xp, double yp, double zp)
{
    /* Variables */
    double r,
           res = 0,
           /* These are used for a dirty little trick to evaluate the primitive
              over the integration limits */
           x[2], y[2], z[2];
    int i, j, k;

    /* First thing to do is make P the origin of the coordinate system */
    x[0] = x1 - xp;
    x[1] = x2 - xp;
    y[0] = y1 - yp;
    y[1] = y2 - yp;
    z[0] = z1 - zp;
    z[1] = z2 - zp;

    /* Do the little summation to evaluate the primitive over the integration
       limits */
    for(k=0; k<=1; k++)
    {
        for(j=0; j<=1; j++)
        {
            for(i=0; i<=1; i++)
            {
                r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);

                res += -pow(-1, i + j + k)*log(z[k] + r);
            }
        }
    }

    /* Now all that is left is to multiply res by the gravitational constant and
       density and convert it to Eotvos units */
    res *= G*SI2EOTVOS*dens;

    return res;
}


/* Calculates the gxz gravity gradient tensor component cause by a prism. */
double prism_gxz(double dens, double x1, double x2, double y1, double y2,
                 double z1, double z2, double xp, double yp, double zp)
{
    /* Variables */
    double r,
           res = 0,
           /* These are used for a dirty little trick to evaluate the primitive
              over the integration limits */
           x[2], y[2], z[2];
    int i, j, k;

    /* First thing to do is make P the origin of the coordinate system */
    x[0] = x1 - xp;
    x[1] = x2 - xp;
    y[0] = y1 - yp;
    y[1] = y2 - yp;
    z[0] = z1 - zp;
    z[1] = z2 - zp;

    /* Do the little summation to evaluate the primitive over the integration
       limits */
    for(k=0; k<=1; k++)
    {
        for(j=0; j<=1; j++)
        {
            for(i=0; i<=1; i++)
            {
                r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);

                res += -pow(-1, i + j + k)*log(y[j] + r);
            }
        }
    }

    /* Now all that is left is to multiply res by the gravitational constant and
       density and convert it to Eotvos units */
    res *= G*SI2EOTVOS*dens;

    return res;
}


/* Calculates the gyy gravity gradient tensor component cause by a prism. */
double prism_gyy(double dens, double x1, double x2, double y1, double y2,
                 double z1, double z2, double xp, double yp, double zp)
{
    /* Variables */
    double r,
           res = 0,
           /* These are used for a dirty little trick to evaluate the primitive
              over the integration limits */
           x[2], y[2], z[2];
    int i, j, k;

    /* First thing to do is make P the origin of the coordinate system */
    x[0] = x1 - xp;
    x[1] = x2 - xp;
    y[0] = y1 - yp;
    y[1] = y2 - yp;
    z[0] = z1 - zp;
    z[1] = z2 - zp;

    /* Do the little summation to evaluate the primitive over the integration
       limits */
    for(k=0; k<=1; k++)
    {
        for(j=0; j<=1; j++)
        {
            for(i=0; i<=1; i++)
            {
                r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);

                /* There should be a minus sign in front of pow but it cancels
                   with the one that should be in front of atan */
                res += pow(-1, i + j + k)*atan2(z[k]*x[i], y[j]*r);
            }
        }
    }

    /* Now all that is left is to multiply res by the gravitational constant and
       density and convert it to Eotvos units */
    res *= G*SI2EOTVOS*dens;

    return res;
}


/* Calculates the gyz gravity gradient tensor component cause by a prism. */
double prism_gyz(double dens, double x1, double x2, double y1, double y2,
                 double z1, double z2, double xp, double yp, double zp)
{
    /* Variables */
    double r,
           res = 0,
           /* These are used for a dirty little trick to evaluate the primitive
              over the integration limits */
           x[2], y[2], z[2];
    int i, j, k;

    /* First thing to do is make P the origin of the coordinate system */
    x[0] = x1 - xp;
    x[1] = x2 - xp;
    y[0] = y1 - yp;
    y[1] = y2 - yp;
    z[0] = z1 - zp;
    z[1] = z2 - zp;

    /* Do the little summation to evaluate the primitive over the integration
       limits */
    for(k=0; k<=1; k++)
    {
        for(j=0; j<=1; j++)
        {
            for(i=0; i<=1; i++)
            {
                r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);

                res += -pow(-1, i + j + k)*log(x[i] + r);
            }
        }
    }

    /* Now all that is left is to multiply res by the gravitational constant and
       density and convert it to Eotvos units */
    res *= G*SI2EOTVOS*dens;

    return res;
}



/* Calculates the gzz gravity gradient tensor component cause by a prism. */
double prism_gzz(double dens, double x1, double x2, double y1, double y2,
                 double z1, double z2, double xp, double yp, double zp)
{
    /* Variables */
    double r,
           res = 0,
           /* These are used for a dirty little trick to evaluate the primitive
              over the integration limits */
           x[2], y[2], z[2];
    int i, j, k;

    /* First thing to do is make P the origin of the coordinate system */
    x[0] = x1 - xp;
    x[1] = x2 - xp;
    y[0] = y1 - yp;
    y[1] = y2 - yp;
    z[0] = z1 - zp;
    z[1] = z2 - zp;

    /* Do the little summation to evaluate the primitive over the integration
       limits */
    for(k=0; k<=1; k++)
    {
        for(j=0; j<=1; j++)
        {
            for(i=0; i<=1; i++)
            {
                r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);

                /* There should be a minus sign in front of pow but it cancels
                   with the one that should be in front of atan */
                res += pow(-1, i + j + k)*atan2(x[i]*y[j], z[k]*r);
            }
        }
    }

    /* Now all that is left is to multiply res by the gravitational constant and
       density and convert it to Eotvos units */
    res *= G*SI2EOTVOS*dens;

    return res;
}
