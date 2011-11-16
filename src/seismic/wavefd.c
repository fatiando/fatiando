/* **************************************************************************

   Time steps for a finite difference solver to the elastic wave equation.

   Author: Leonardo Uieda
   Date: 29 June 2010

   ************************************************************************** */

#include <stdio.h>
#include "wavefd.h"



int timestep1d(double deltax, double deltat, double *u_tm1, double *u_t,
        double *vel, int size, double *u_tp1)
{
    int k;
    double *pu_tp1_k, *pu_t_kp1, *pu_t_k, *pu_t_km1, *pu_tm1_k, *pvel;
    double deltat_sqr, deltax_sqr;

    deltat_sqr = deltat*deltat;

    deltax_sqr = deltax*deltax;

    pvel = vel + 1;

    pu_tp1_k = u_tp1 + 1;

    pu_t_km1 = u_t;

    pu_t_k = u_t + 1;

    pu_t_kp1 = u_t + 2;

    pu_tm1_k = u_tm1 + 1;

    for(k=1; k < (size - 1); k++)
    {
        *pu_tp1_k = (deltat_sqr)*((*pvel)*(*pvel))*(
                    (*pu_t_kp1) - 2*(*pu_t_k) + (*pu_t_km1)
                )/(deltax_sqr) + 2*(*pu_t_k) - (*pu_tm1_k);

        pvel++;

        pu_tp1_k++;

        pu_t_km1++;

        pu_t_k++;

        pu_t_kp1++;

        pu_tm1_k++;
    }
    return size;
}
