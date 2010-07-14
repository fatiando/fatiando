/* **************************************************************************

   This module contains special function implementations such as the error
   function, the complementary error function (and its derivative), etc.

   Author: Leonardo Uieda
   Date: 12 Jul 2010

   ************************************************************************** */


#ifndef _SPECIAL_H_
#define _SPECIAL_H_



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
extern double FAT_erf(double x, double power_coef, double mult_coef, int ncells);


extern double FAT_erfc_deriv(double x);

#endif
