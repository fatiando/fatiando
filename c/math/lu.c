/* **************************************************************************

   Set of functions for performing the LU decomposition of a matrix.

   OBS: All matrices are treated as 1-dimensional double arrays and are accessed
        using the macro POS in lu.h

   Author: Leonardo Uieda
   Date: 01 March 2010

   ************************************************************************** */

#include <stdlib.h>
#include "lu.h"

/* LU_DECOMP_NOPIVOT */
/* ************************************************************************** */
/* Does the LU decomposition of a square matrix of dimensions 'dim' WITHOUT
   pivoting. Returns the result in the previously allocated matrix 'lu'. Matrix
   U is given in the upper triangle and L in the lower. The diagonal of L (=1)
   is omitted. REMEMBER: pre-allocate the memory for the buffers!
   Parameters:
       matrix: pointer to a 1D array containing a SQUARE matrix such as
               [a11,a12,a13,...,a1N,a21,a22,...,a2N,...,aNN]
       dim: the dimension N of the matrix
       lu: the'dim'x'dim' sized buffer where the L and U matrices will be
           returned.
   Returns:
       'dim' if all went well;
       0 if the matrix is not decomposable

   NOTE: The pivoting alternative to this function is prefered because it was
   more extensively tested. See lu_decomp_pivot documentation for more details.
*/
int lu_decomp_nopivot(double *matrix, int dim, double *lu)
{
    /* Variables */
    int i, j, k; /* Position counters */

    /* Iterate over the lines */
    for(i=0; i<dim; i++)
    {
        /* Calculate the terms of the ith line of L */
        for(j=0; j<i; j++)
        {
            lu[POS(i,j,dim)] = matrix[POS(i,j,dim)];

            for(k=0; k<j; k++)
            {
                /* l[i,j]               l[i,k]           u[k,j] */
                lu[POS(i,j,dim)] -= lu[POS(i,k,dim)]*lu[POS(k,j,dim)];
            }

            /* Check if u[j,j]!=0 */
            if(lu[POS(j,j,dim)]!=0)
            {
                lu[POS(i,j,dim)] = (double)lu[POS(i,j,dim)]/lu[POS(j,j,dim)];
            }
            else
            {
                return 0;
            }
        }

        /* Calculate the terms of the ith line of U.
           (Starts in i because there are no elements under the diagonal) */
        for(j=i; j<dim; j++)
        {
            lu[POS(i,j,dim)] = matrix[POS(i,j,dim)];

            for(k=0; k<i; k++)
            {
                /* u[i,j]               l[i,k]           u[k,j] */
                lu[POS(i,j,dim)] -= lu[POS(i,k,dim)]*lu[POS(k,j,dim)];
            }
        }

    }

    return dim;
}
/* ************************************************************************** */


/* LU_DECOMP */
/* ************************************************************************** */
/* Does the LU decomposition of a square matrix of dimensions 'dim' using
   pivoting. Returns the result in the previously allocated vector 'lu'. Matrix
   U is given in the upper triangle and L in the lower. The diagonal of L (=1)
   is omitted. The permutations are returned in 'permut' (line i was permuted
   with line permut[i]). REMEMBER: pre-allocate the memory for the buffers!
   Parameters:b[i]
       matrix: pointer to a 1D array containing a SQUARE matrix such as
               [a11,a12,a13,...,a1N,a21,a22,...,a2N,...,aNN]
       dim: the dimension N of the matrix
       lu: the buffer where the L and U matrices will be returned. The diagonal
           elements of L are omitted (=1) and the 2 matrices are joined into one
       permut: the 'dim' sized buffer where the permutations will be returned.
   Returns:
       'dim' if all went well;
       0 if the matrix is not decomposable
*/
int lu_decomp(double *matrix, int dim, double *lu, int *permut)
{
    int i, j, k, pivot_index;
    double maximum, tmp;

    /* Copy matrix into lu so that I can do the permutations without messing
       up the contents of matrix */
    for(i=0; i<dim; i++)
    {
        for(j=0; j<dim; j++)
        {
            lu[POS(i,j,dim)] = matrix[POS(i,j,dim)];
        }
    }

    /* Iterate over the lines of U and columns of L */
    for(i=0; i<dim; i++)
    {
        /* Calculate Uii and the ith column of L (without dividing by Ujj)
           j is now the line index and starts at j=i in order to calculate Uii
           (or in this case, Ujj). Also, take note of the maximum value in this
           column */
        maximum = 0;
        pivot_index = -1;
        for(j=i; j<dim; j++)
        {
            /* Lji = Aji is not needed because it is already there when I copied
               the matrix into lu */

            /* Do the summation */
            for(k=0; k<i; k++)
            {
                /*    Lji                Ljk                Uki       */
                lu[POS(j,i,dim)] -= lu[POS(j,k,dim)]*lu[POS(k,i,dim)];
            }

            /* Check if the new value is bigger than the biggest so far */
            if(ABSOLUTE(lu[POS(j,i,dim)]) > maximum)
            {
                /* Update the maximum */
                maximum = ABSOLUTE(lu[POS(j,i,dim)]);

                /* Mark the line of the biggest value */
                pivot_index = j;
            }
        }

        /* If none of the values is bigger than 0 (i.e. pivot_index is still -1)
           Return an error message because th matrix can't be decomposed. */
        if(pivot_index == -1)
        {
            return 0;
        }

        /* Mark in permut[i] which line will line i be permuted with */
        permut[i] = pivot_index;

        /* Now that we know the pivot, do the permutation of the pivot line with
           the ith line (but only if the pivot is not the ith line) */
        if(pivot_index != i)
        {
            /* Swap the lines of the LU matrix (after the diagonal the elements
               are of 'matrix'. So it's lines get swapped as well). */
            for(j=0; j<dim; j++)
            {
                tmp = lu[POS(i,j,dim)];
                lu[POS(i,j,dim)] = lu[POS(pivot_index,j,dim)];
                lu[POS(pivot_index,j,dim)] = tmp;
            }
        }

        /* Now that the lines have been permuted I can divide Lji by Uii */
        for(j=i+1; j<dim; j++)
        {
            lu[POS(j,i,dim)] = (double) lu[POS(j,i,dim)]/lu[POS(i,i,dim)];
        }


        /* Finally, calculate the ith line of U (excluding Uii that was already
           calculated earlier). Now j is the column index. */
        for(j=i+1; j<dim; j++)
        {
            /* Uij = Aij is not needed because it is already there when I copied
               the matrix into lu */

            /* Do the summation */
            for(k=0; k<i; k++)
            {
                /*    Uij                Lik                Ukj       */
                lu[POS(i,j,dim)] -= lu[POS(i,k,dim)]*lu[POS(k,j,dim)];
            }
        }
    }

    /* If it all went well, return the dimension of the matrices */
    return dim;
}
/* ************************************************************************** */


/* SOLVE_LU */
/* ************************************************************************** */
/* Solve a linear system Ax=y given it's LU decomposition (with pivoting).
 * REMEMBER: pre-allocate the memory for the buffers!
 * Parameters:
 *      lu: pointer to a 1D array containing the LU decomposition of a matrix
 *          such as returned by lu_decomp
 *          EX: [u11,u12,u13,...,u1N,l21,u22,...,u2N,...,uNN]
 *      dim: the dimension N of the matrix
 *      permut: pointer to a 1D array with the permutations done in the LU
 *         decomposition as returned by lu_decomp
 *      y: pointer to a 1D array with the ordinate values;
 *      x: pointer to a 1D array with the output buffer
 *  Returns:
 *      'dim' if all went well
 *      0 if an error occurred
 */
int solve_lu(double *lu, int dim, int *permut, double *y, double *x)
{
    int i, k;
    double *b, tmp;

    /* Divide the system LUx=y into Lb=y and Ux=b */

    /* First malloc memory for b */
    b = (double *)malloc(dim*sizeof(double));
    if(!b)
    {
        return 0;
    }

    /* First, copy y into b so that I can permute y in place. The initial value
     * for b should be y anyway. */
    for(i=0; i<dim; i++)
    {
        b[i] = y[i];
    }

    /* Permute b */
    for(i=0; i<dim; i++)
    {
		tmp = b[i];
		b[i] = b[permut[i]];
		b[permut[i]] = tmp;
    }

    /* Now solve the first system for b */
    for(i=0; i<dim; i++)
    {
        /* By now b[i] is already y_permuted[i] */

        for(k=0; k<i; k++)
        {
            b[i] -= lu[POS(i,k,dim)]*b[k];
        }
    }

    /* Now that I have b, solve the second system for x by back propagating */
    for(i=dim-1; i>=0; i--)
    {
        x[i] = b[i];

        for(k=i+1; k<dim; k++)
        {
            x[i] -= x[k]*lu[POS(i,k,dim)];
        }

        x[i] = (double) x[i]/lu[POS(i,i,dim)];
    }

    /* Free the memory for b */
    free(b);

    /* Return the dimension */
    return dim;
}
/* ************************************************************************** */


/* INV_LU */
/* ************************************************************************** */
/* Calculate the inverse of a square matrix given it's LU decomposition
 * (with pivoting).
 * REMEMBER: pre-allocate the memory for the buffers!
 * Parameters:
 *      lu: pointer to a 1D array containing the LU decomposition of a matrix
 *          such as returned by lu_decomp
 *          EX: [u11,u12,u13,...,u1N,l21,u22,...,u2N,...,uNN]
 *      dim: the dimension N of the matrix
 *      permut: pointer to a 1D array with the permutations done in the LU
 *         decomposition as returned by lu_decomp
 *      inv: pointer to a 1D array matrix for the output buffer;
 *  Returns:
 *      'dim' if all went well
 *      0 if an error occurred
 */
int inv_lu(double *lu, int dim, int *permut, double *inv)
{
    int i, k, j;
    double *b, tmp;

    /* Divide the system LU(invA)=I dim number of systems and solve for each
     * column of invA. The systems are solved like in solve_lu */

    /* First malloc memory for b */
    b = (double *)malloc(dim*sizeof(double));
    if(!b)
    {
        return 0;
    }

    /* Calculate each column of inv */
    for(j=0; j<dim; j++)
    {
        /* First, copy the jth column of I into b so that I can permute it in
         * place. That should be the initial value for b anyway. */
        for(i=0; i<dim; i++)
        {
            b[i] = 0;
        }

        b[j] = 1;

        /* Permute b */
        for(i=0; i<dim; i++)
        {
            tmp = b[i];
            b[i] = b[permut[i]];
            b[permut[i]] = tmp;
        }

        /* Now solve the first system for b */
        for(i=0; i<dim; i++)
        {
            /* By now b[i] is already y_permuted[i] */

            for(k=0; k<i; k++)
            {
                b[i] -= lu[POS(i,k,dim)]*b[k];
            }
        }

        /* Now that we have b, solve the second system for jth column of the
         * inverse by back propagating */
        for(i=dim-1; i>=0; i--)
        {
            inv[POS(i,j,dim)] = b[i];

            for(k=i+1; k<dim; k++)
            {
                inv[POS(i,j,dim)] -= inv[POS(k,j,dim)]*lu[POS(i,k,dim)];
            }

            inv[POS(i,j,dim)] = (double) inv[POS(i,j,dim)]/lu[POS(i,i,dim)];
        }
    }

    /* Free the memory for b */
    free(b);

    /* Return the dimension */
    return dim;
}
/* ************************************************************************** */
