/* **************************************************************************

   Set of functions for numerical integration via Gauss-Legendre Quadrature
   (GLQ).

   To integrate f(x) in the interval [a,b] using the GLQ:

       1) choose an order N for the quadrature (points in which f(x) will be
          discretized

       2) calculate the nodes using
           glq_nodes(N, my_nodes);

       3) calculate the weights using
           glq_weights(N, my_nodes, my_weights);

       4) scale the nodes to the interval [a,b] using
           glq_scale_nodes(a, b, N, my_nodes, my_scaled_nodes);

       5) do the summation of the weighted f(my_scaled_nodes)
           result = 0;
           for(i=0; i < N; i++)
           {
               result += my_weights[i]*f(my_scaled_nodes[i]);
           }
           /* Account for the change in variables done when scaling the nodes
           result *= (b - a)/2;


   Author: Leonardo Uieda
   Date: 07 May 2010

   ************************************************************************** */

#include <math.h>
#include "glq.h"


/* GLQ_NODES */
/* ************************************************************************** */
/* Calculates the GLQ nodes. The nodes are the roots of the Legendre polynomial
   of degree 'order'. The roots are calculated using Newton's Method for
   Multiple Roots (Barrera-Figueroa et al., 2006).

   REMEMBER: pre-allocate the memory for the buffer!

   Parameters:
       order: order of the GLQ
       nodes: output buffer for the nodes

   Returns:
       'order' if all went well
       0 if there was stagnation when looking for the roots
*/
int glq_nodes(int order, double *nodes)
{
    double x1, x0, pn_2, pn_1, pn, pn_line, rootsum;
    int i, n, iterations;

    for(i=0; i < order; i++)
    {
        /* Set the initial guess for this root */
        x1 = cos((double)PI*(i + 0.75)/(order + 0.5));

        for(iterations=0; iterations < GLQ_MAXIT; iterations++)
        {
            x0 = x1;

            /* Need to find Pn(xi) and Pn'(xi):
               To do this, use the recursive relation to find Pn and Pn-1:
                 Pn(x) = (2n-1)xPn_1(x)/n - (n-1)Pn_2(x)/n
               Then use:
                 Pn'(x) = n(xPn(x)-Pn_1(x))/(x*x-1) */

            /* Find Pn and Pn-1 stating from P0 and P1 */
            pn_1 = 1.0;       /* This is Po(x) */
            pn = x0;          /* and this P1(x) */
            for(n=2; n <= order; n++)
            {
                pn_2 = pn_1;
                pn_1 = pn;
                pn = ((2*n - 1)*x0*pn_1 - (n - 1)*pn_2) / n;
            }

            /* Now find Pn'(xi) */
            pn_line = (double) order*(x0*pn - pn_1)/(x0*x0 - 1);

            /* Sum the roots found so far */
            rootsum = 0;
            for(n=0; n < i; n++)
            {
                rootsum += (double) 1/(x0 - nodes[n]);
            }

            /* Update the guess for the root */
            x1 = x0 - (double) pn/(pn_line - pn*rootsum);

            if(ABSOLUTE(x1 - x0) <= GLQ_MAXERROR)
            {
                break;
            }
        }

        /* Return 0 if there was stagnation */
        if(iterations >= GLQ_MAXIT && ABSOLUTE(x1 - x0) > GLQ_MAXERROR)
        {
            return 0;
        }
        else
        {
            /* Put the value of the root found in the nodes */
            nodes[i] = x1;
        }
    }

    return order;
}


/* GLQ_WEIGHTS */
/* ************************************************************************** */
/* Calculates the weights for the GLQ.

   REMEMBER: pre-allocate the memory for the buffer!

   Parameters:
       order: order of the GLQ
       nodes: pointer to array with the UN-SCALED nodes
       weights: output buffer for the weights

   Returns:
       'order' if all went well
       0 if fails
*/
int glq_weights(int order, double *nodes, double *weights)
{
    double pn_2, pn_1, pn, pn_line;
    int i, n;

    for(i=0; i < order; i++)
    {
        /* Need to find Pn'(xi):
           To do this, use the recursive relation to find Pn and Pn-1:
             Pn(x) = (2n-1)xPn_1(x)/n - (n-1)Pn_2(x)/n
           Then use:
             Pn'(x) = n(xPn(x)-Pn_1(x))/(x*x-1) */

        /* Find Pn and Pn-1 stating from P0 and P1 */
        pn_1 = 1.0;       /* This is Po(x) */
        pn = nodes[i];    /* and this P1(x) */
        for(n=2; n <= order; n++)
        {
            pn_2 = pn_1;
            pn_1 = pn;
            pn = ((2*n - 1)*nodes[i]*pn_1 - (n - 1)*pn_2) / n;
        }

        /* Now find Pn'(xi) */
        pn_line = (double) order*(nodes[i]*pn - pn_1)/(nodes[i]*nodes[i] - 1);

        /* Calculate the weight Wi */
        weights[i] = (double) 2/((1 - nodes[i]*nodes[i])*(pn_line*pn_line));
    }

    return order;
}


/* GLQ_SCALE_NODES */
/* ************************************************************************** */
/* Scales the GLQ nodes to the integration interval [a,b]. This is necessary
   because the output of glq_nodes is scaled to [-1,1] (which is the value
   gle_weights needs).

   REMEMBER: pre-allocate the memory for the buffer!

   Parameters:
       a: lower integration limit
       b: upper integration limit
       order: order of the GLQ
       nodes: pointer to array with the UN-SCALED nodes
       scaled: output buffer for the scaled nodes

   Returns:
       'order' if all went well
       0 if fails
*/
int glq_scale_nodes(double a, double b, int order, double *nodes,
                    double *scaled)
{
    double tmpplus, tmpminus;
    int i;

    tmpplus = 0.5*(b + a);
    tmpminus = 0.5*(b - a);

    for(i=0; i<order; i++)
    {
        scaled[i] = tmpminus*nodes[i] + tmpplus;
    }

    return order;
}
