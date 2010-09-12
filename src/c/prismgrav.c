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

   This module contains a set of functions that calculate the gravitational
   potential and its first and second derivatives for the rectangular prism
   using the formulas in Nagy (2000).

   Author: Leonardo Uieda
   Date: 01 March 2010

   ************************************************************************** */

#include <math.h>
#include "prismgrav.h"


/* Calculates the gz gravity component cause by a prism. */
double prism_gz(double dens, double x1, double x2, double y1, double y2,
                double z1, double z2, double xp, double yp, double zp)
{
    /* Variables */
    double r,
           res,
           deltax1, deltax2, deltay1, deltay2, deltaz1, deltaz2;

    /* First thing to do is make P the origin of the coordinate system */
    deltax1 = x1 - xp;
    deltax2 = x2 - xp;
    deltay1 = y1 - yp;
    deltay2 = y2 - yp;
    deltaz1 = z1 - zp;
    deltaz2 = z2 - zp;

    res = 0;

    /* Evaluate the integration limits */
    r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1);

    res += 1*(deltax1*log(deltay1 + r) + deltay1*log(deltax1 + r) -
    		deltaz1*atan2(deltax1*deltay1, deltaz1*r));

    r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1);

    res += -1*(deltax2*log(deltay1 + r) + deltay1*log(deltax2 + r) -
    		deltaz1*atan2(deltax2*deltay1, deltaz1*r));

    r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1);

    res += -1*(deltax1*log(deltay2 + r) + deltay2*log(deltax1 + r) -
    		deltaz1*atan2(deltax1*deltay2, deltaz1*r));

    r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1);

    res += 1*(deltax2*log(deltay2 + r) + deltay2*log(deltax2 + r) -
    		deltaz1*atan2(deltax2*deltay2, deltaz1*r));

    r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2);

    res += -1*(deltax1*log(deltay1 + r) + deltay1*log(deltax1 + r) -
    		deltaz2*atan2(deltax1*deltay1, deltaz2*r));

    r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2);

    res += 1*(deltax2*log(deltay1 + r) + deltay1*log(deltax2 + r) -
    		deltaz2*atan2(deltax2*deltay1, deltaz2*r));

    r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2);

    res += 1*(deltax1*log(deltay2 + r) + deltay2*log(deltax1 + r) -
    		deltaz2*atan2(deltax1*deltay2, deltaz2*r));

    r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2);

    res += -1*(deltax2*log(deltay2 + r) + deltay2*log(deltax2 + r) -
    		deltaz2*atan2(deltax2*deltay2, deltaz2*r));

    /* Now all that is left is to multiply res by the gravitational constant and
       density and convert it to mGal units */
    res *= G*SI2MGAL*dens;

    return res;
}


/* Calculates the gxx gravity gradient tensor component cause by a prism. */
double prism_gxx(double dens, double x1, double x2, double y1, double y2,
                double z1, double z2, double xp, double yp, double zp)
{
    /* Variables */
	double r,
		   res,
		   deltax1, deltax2, deltay1, deltay2, deltaz1, deltaz2;

	/* First thing to do is make P the origin of the coordinate system */
	deltax1 = x1 - xp;
	deltax2 = x2 - xp;
	deltay1 = y1 - yp;
	deltay2 = y2 - yp;
	deltaz1 = z1 - zp;
	deltaz2 = z2 - zp;

	res = 0;

	/* Evaluate the integration limits */
	r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1);

	res += 1*atan2(deltay1*deltaz1, deltax1*r);

	r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1);

	res += -1*atan2(deltay1*deltaz1, deltax2*r);

	r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1);

	res += -1*atan2(deltay2*deltaz1, deltax1*r);

	r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1);

	res += 1*atan2(deltay2*deltaz1, deltax2*r);

	r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2);

	res += -1*atan2(deltay1*deltaz2, deltax1*r);

	r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2);

	res += 1*atan2(deltay1*deltaz2, deltax2*r);

	r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2);

	res += 1*atan2(deltay2*deltaz2, deltax1*r);

	r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2);

	res += -1*atan2(deltay2*deltaz2, deltax2*r);

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
		   res,
		   deltax1, deltax2, deltay1, deltay2, deltaz1, deltaz2;

	/* First thing to do is make P the origin of the coordinate system */
	deltax1 = x1 - xp;
	deltax2 = x2 - xp;
	deltay1 = y1 - yp;
	deltay2 = y2 - yp;
	deltaz1 = z1 - zp;
	deltaz2 = z2 - zp;

	res = 0;

	/* Evaluate the integration limits */
    r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1);

    res += 1*(-1*log(deltaz1 + r));

    r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1);

    res += -1*(-1*log(deltaz1 + r));

    r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1);

    res += -1*(-1*log(deltaz1 + r));

    r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1);

    res += 1*(-1*log(deltaz1 + r));

    r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2);

    res += -1*(-1*log(deltaz2 + r));

    r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2);

    res += 1*(-1*log(deltaz2 + r));

    r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2);

    res += 1*(-1*log(deltaz2 + r));

    r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2);

    res += -1*(-1*log(deltaz2 + r));

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
		   res,
		   deltax1, deltax2, deltay1, deltay2, deltaz1, deltaz2;

	/* First thing to do is make P the origin of the coordinate system */
	deltax1 = x1 - xp;
	deltax2 = x2 - xp;
	deltay1 = y1 - yp;
	deltay2 = y2 - yp;
	deltaz1 = z1 - zp;
	deltaz2 = z2 - zp;

	res = 0;

	/* Evaluate the integration limits */
	r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1);

	res += 1*(-1*log(deltay1 + r));

	r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1);

	res += -1*(-1*log(deltay1 + r));

	r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1);

	res += -1*(-1*log(deltay2 + r));

	r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1);

	res += 1*(-1*log(deltay2 + r));

	r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2);

	res += -1*(-1*log(deltay1 + r));

	r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2);

	res += 1*(-1*log(deltay1 + r));

	r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2);

	res += 1*(-1*log(deltay2 + r));

	r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2);

	res += -1*(-1*log(deltay2 + r));

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
		   res,
		   deltax1, deltax2, deltay1, deltay2, deltaz1, deltaz2;

	/* First thing to do is make P the origin of the coordinate system */
	deltax1 = x1 - xp;
	deltax2 = x2 - xp;
	deltay1 = y1 - yp;
	deltay2 = y2 - yp;
	deltaz1 = z1 - zp;
	deltaz2 = z2 - zp;

	res = 0;

	/* Evaluate the integration limits */
	r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1);

	res += 1*(atan2(deltaz1*deltax1, deltay1*r));

	r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1);

	res += -1*(atan2(deltaz1*deltax2, deltay1*r));

	r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1);

	res += -1*(atan2(deltaz1*deltax1, deltay2*r));

	r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1);

	res += 1*(atan2(deltaz1*deltax2, deltay2*r));

	r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2);

	res += -1*(atan2(deltaz2*deltax1, deltay1*r));

	r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2);

	res += 1*(atan2(deltaz2*deltax2, deltay1*r));

	r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2);

	res += 1*(atan2(deltaz2*deltax1, deltay2*r));

	r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2);

	res += -1*(atan2(deltaz2*deltax2, deltay2*r));

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
		   res,
		   deltax1, deltax2, deltay1, deltay2, deltaz1, deltaz2;

	/* First thing to do is make P the origin of the coordinate system */
	deltax1 = x1 - xp;
	deltax2 = x2 - xp;
	deltay1 = y1 - yp;
	deltay2 = y2 - yp;
	deltaz1 = z1 - zp;
	deltaz2 = z2 - zp;

	res = 0;

	/* Evaluate the integration limits */
    r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1);

    res += 1*(-1*log(deltax1 + r));

    r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1);

    res += -1*(-1*log(deltax2 + r));

    r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1);

    res += -1*(-1*log(deltax1 + r));

    r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1);

    res += 1*(-1*log(deltax2 + r));

    r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2);

    res += -1*(-1*log(deltax1 + r));

    r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2);

    res += 1*(-1*log(deltax2 + r));

    r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2);

    res += 1*(-1*log(deltax1 + r));

    r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2);

    res += -1*(-1*log(deltax2 + r));

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
		   res,
		   deltax1, deltax2, deltay1, deltay2, deltaz1, deltaz2;

	/* First thing to do is make P the origin of the coordinate system */
	deltax1 = x1 - xp;
	deltax2 = x2 - xp;
	deltay1 = y1 - yp;
	deltay2 = y2 - yp;
	deltaz1 = z1 - zp;
	deltaz2 = z2 - zp;

	res = 0;

	/* Evaluate the integration limits */
    r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1);

    res += 1*(atan2(deltax1*deltay1, deltaz1*r));

    r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1);

    res += -1*(atan2(deltax2*deltay1, deltaz1*r));

    r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1);

    res += -1*(atan2(deltax1*deltay2, deltaz1*r));

    r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1);

    res += 1*(atan2(deltax2*deltay2, deltaz1*r));

    r = sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2);

    res += -1*(atan2(deltax1*deltay1, deltaz2*r));

    r = sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2);

    res += 1*(atan2(deltax2*deltay1, deltaz2*r));

    r = sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2);

    res += 1*(atan2(deltax1*deltay2, deltaz2*r));

    r = sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2);

    res += -1*(atan2(deltax2*deltay2, deltaz2*r));

    /* Now all that is left is to multiply res by the gravitational constant and
       density and convert it to Eotvos units */
    res *= G*SI2EOTVOS*dens;

    return res;
}
