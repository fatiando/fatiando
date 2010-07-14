/* **************************************************************************

   This module contains special function implementations such as the error
   function, the complementary error function (and its derivative), etc.

   Author: Leonardo Uieda
   Date: 12 Jul 2010

   ************************************************************************** */

#include <math.h>
#include "special.h"


/* FAT_ERF(x, ncells)
 * Calculate the error function using a rectangular rule for numerical
 * integration.
 *
 * Parameters:
 *
 * 		x: variable at which to evaluate erf
 *
 * 		ncells: into how many intervals the integration region will be split
 */
double FAT_erf(double x, double power_coef, double mult_coef, int ncells)
{
	double res, step, xsi;
	register int i;

	step = (double) x/ncells;

    res = 0.0;

    for(i=0; i < ncells; i++)
	{
    	xsi = 0.5*step + i*step;

        res += mult_coef*exp(-(power_coef*xsi*xsi))*step;
	}

    /* 2/sqrt(pi) = 1.1283791670955126 */
    res *= 1.1283791670955126;

    return res;
}


double FAT_erfc_deriv(double x)
{
	/* 2/sqrt(pi) = 1.1283791670955126 */
	return -1.1283791670955126*exp(-(x*x));
}
