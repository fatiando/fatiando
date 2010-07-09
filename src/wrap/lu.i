/* **************************************************************************
   Interface file for generation SWIG wrappers around 'lu.c'

   OBS: Be careful with the typemaps! Using different types with the same
        argument names will cause wierd problems! 
   ************************************************************************** */

/* The module docstring */
%define DOCSTRING
"
LU:
    C-coded set of functions for calculating the LU decomposition of a matrix,
    solving linear systems and calculating the inverse of a matrix.

Author: Leonardo Uieda
Created 01 March 2010
"
%enddef
/* Declare the module name */
%module(docstring=DOCSTRING) lu


/* Put the headers with the definitions */
%{

#include "../c/math/lu.h"
#include "typeconversions.c"

%}


/* LU_DECOMP_NOPIVOT */
/* ************************************************************************** */
/* The in typemap to recieve a 2D python list (matrix) and convert ir to a C 1D
   array representing a square matrix.
   *lu is the 'dim'x'dim' sized buffer where the LU decomposition will be
   returned. */
%typemap(in) (double *matrix, int dim, double *lu)
{
    /* Check and convert the input to a C array */
    $1 = PyList_to_square_matrix($input, &$2);

    /* Check if there was any error in the conversion */
    if(!$1)
    {
        return NULL;
    }

    /* Finally, malloc some memory for the output buffers */
    $3 = (double *)malloc($2*$2*sizeof(double));
}

/* This cleans up the array we malloc'd before the function call */
%typemap(freearg) (double *matrix, int dim, double *lu)
{
    if($1)
    {
        free($1);
    }
}

/* Convert the temp output buffer to a 2D Python list (the dimension will come
   in the result) */
%typemap(argout) (double *matrix, int dim, double *lu)
{
    /* First, check if there was any problem with the decomposition */
    if(!result)
    {
        /* Liberate the memory used for the output buffer */
        free($3);
        /* Raise the appropriate exception */
        PyErr_SetString(PyExc_AttributeError, "Cannot perform LU decomposition of the given matrix.");
        return NULL;
    }

    /* Blow away any previous result */
    Py_XDECREF($result);

    /* Convert the C array to a Python List */
    $result = square_matrix_to_PyList($3, result);

    /* Liberate the memory used for the output buffer */
    free($3);
}

/* EXPOSE THE LU_DECOMP_NOPIVOT FUNCTION */
%feature("autodoc", "1");
%define LU_DECOMP_NOPIVOTDOC
"
Does the LU decomposition of a square matrix WITHOUT pivoting. The decomposition is:

    A = LU

Returns both L and U matrices in a single 2D list (omits the diagonal of L because it is equal to 1).
The input matrix is a 2D python list object containing a SQUARE matrix.

Ex:

    lu = lu.decomp_nopivot(matrix)


NOTE: The pivoting alternative to this function is prefered because it was more extensively tested. See lu.decomp documentation for more details.
"
%enddef

%feature("docstring", LU_DECOMP_NOPIVOTDOC);
%rename(decomp_nopivot) lu_decomp_nopivot;
extern int lu_decomp_nopivot(double *matrix, int dim, double *lu);

/* ************************************************************************** */


/* LU_DECOMP */
/* ************************************************************************** */
/* The in typemap to recieve a 2D python list (matrix) and convert ir to a C 1D
   array representing a square matrix.
   *lu is the 'dim'x'dim' sized buffer where the LU decomposition will be
   returned.
   *permut is the 'dim' sized buffer where the permutations will be returned */
%typemap(in) (double *matrix, int dim, double *lu, int *permut)
{
    /* Check and convert the input to a C array */
    $1 = PyList_to_square_matrix($input, &$2);

    /* Check if there was any error in the conversion */
    if(!$1)
    {
        return NULL;
    }

    /* Finally, malloc some memory for the output buffers */
    $3 = (double *)malloc($2*$2*sizeof(double));
    $4 = (int *)malloc($2*sizeof(int));
}

/* This cleans up the array we malloc'd before the function call */
%typemap(freearg) (double *matrix, int dim, double *lu, int *permut)
{
    if($1)
    {
        free($1);
    }
}

/* Convert the temp output buffers to Python list and return them both
   (the dimension will come in the result) */
%typemap(argout) (double *matrix, int dim, double *lu, int *permut)
{
    /* First, check if there was any problem with the decomposition */
    if(!result)
    {
        /* Liberate the memory used for the output buffers */
        free($3);
        free($4);
        /* Raise the appropriate exception */
        PyErr_SetString(PyExc_AttributeError, "Cannot perform LU decomposition of the given matrix.");
        return NULL;
    }

    /* Blow away any previous result */
    Py_XDECREF($result);

    PyObject *lumatrix, *pvector;

    /* Create the new list object to be able to return both the LU matrix and
       the permutations vector */
    $result = PyList_New(2);

    /* Convert LU to a PyList */
    lumatrix = square_matrix_to_PyList($3, result);

    /* Put the LU matrix in the return list */
    PyList_SetItem($result, 0, lumatrix);

    /* Convert the permut vector to a PyList */
    pvector = int_vector_to_PyList($4, result);

    /* Put the permut vector in the return list */
    PyList_SetItem($result, 1, pvector);

    /* Liberate the memory used for the output buffers */
    free($3);
    free($4);
}

/* EXPOSE THE LU_DECOMP FUNCTION */
%feature("autodoc", "1");
%define LU_DECOMPDOC
"
Does the LU decomposition of a square matrix with pivoting. The decomposition is:

    PA = LU  (in this case, P is a permutation matrix. This function returns it in a vector)

Returns both L and U matrices in a single 2D list (omits the diagonal of L because it is equal to 1), and the permutations in the for of a 1D list.
The permutations are in the form:

    line i was permuted with line p[i]

The input matrix is a 2D python list object containing a SQUARE matrix.

Ex:

    lu, p = lu.decomp(matrix)

NOTE: For a non-pivoting version of lu.decomp, see docs for lu.decomp_nopivot
"
%enddef

%feature("docstring", LU_DECOMPDOC);
%rename(decomp) lu_decomp;
extern int lu_decomp(double *matrix, int dim, double *lu, int *permut);

/* ************************************************************************** */


/* SOLVE_LU */
/* ************************************************************************** */

/* A temp variable used to check if the dimension of the input matrix A is the
   same as the input vector y */
%{
int lu_tmpdim;
%}


/* The in typemap to recieve a 2D python list (matrix) and convert ir to a C 1D
   array representing a square matrix. */
%typemap(in) (double *lu, int dim)
{
    /* Check and convert the input to a C array */
    $1 = PyList_to_square_matrix($input, &$2);

    /* Check if there was any error in the conversion */
    if(!$1)
    {
        return NULL;
    }

    /* Set the tmpdim */
    lu_tmpdim = $2;
}
/* This cleans up the array we malloc'd before the function call */
%typemap(freearg) (double *lu, int dim)
{
    if($1)
    {
        free($1);
    }
}

/* In typemap for the permutation array */
%typemap(in) (int *permut)
{
    /* A tmp variable to hold the size of the PyList recieved */
    int tmpsize;

    /* Check and convert the input to a C array */
    $1 = PyList_to_int_vector($input, &tmpsize);

    /* Check if there was any error in the conversion */
    if(!$1)
    {
        return NULL;
    }

    /* Check if the size of the list is the same as the dimension of the system
       matrix A */
    if(tmpsize != lu_tmpdim)
    {
        /* Raise the appropriate exception */
        PyErr_SetString(PyExc_AttributeError, "Permutations list must have the same number of elements as A has lines.");
        free($1);
        return NULL;
    }
}
/* This cleans up the array we malloc'd before the function call */
%typemap(freearg) (int *permut)
{
    if($1)
    {
        free($1);
    }
}


/* The in typemap to recieve a 1D python list (vector) and convert ir to a C 1D
   array and allocate the memory for the output buffer vector. */
%typemap(in) (double *y, double *x)
{
    /* A tmp variable to hold the size of the PyList recieved */
    int tmpsize;

    /* Check and convert the input to a C array */
    $1 = PyList_to_vector($input, &tmpsize);

    /* Check if there was any error in the conversion */
    if(!$1)
    {
        return NULL;
    }

    /* Check if the size of the list is the same as the dimension of the system
       matrix A */
    if(tmpsize != lu_tmpdim)
    {
        /* Raise the appropriate exception */
        PyErr_SetString(PyExc_AttributeError, "In the linear system Ax=y, y list must have the same number of elements as A has lines.");
        free($1);
        return NULL;
    }

    /* Now allocate the memory for the output buffer */
    $2 = (double *)malloc(tmpsize*sizeof(double));
}
/* This cleans up the array we malloc'd before the function call */
%typemap(freearg) (double *y, double *x)
{
    if($1)
    {
        free($1);
    }
}

/* Convert the temp output buffer to a Python list (the dimension will come
   in the result) */
%typemap(argout) (double *lu, int dim, int *permut, double *y, double *x)
{
    /* First, check if there was any problem */
    if(!result)
    {
        /* Liberate the memory used for the output buffer */
        free($5);
        /* Raise the appropriate exception */
        PyErr_SetString(PyExc_MemoryError, "Memory allocation problem.");
        return NULL;
    }

    /* Blow away any previous result */
    Py_XDECREF($result);

    /* Convert the C vector to a PyList */
    $result = vector_to_PyList($5, result);

    /* Liberate the memory used for the output buffer */
    free($5);
}

/* EXPOSE THE SOLVE_LU FUNCTION */
%feature("autodoc", "1");
%define SOLVE_LUDOC
"
Solve a linear system Ax=y given it's LU decomposition (with pivoting).
For LU decomposition, see function 'lu_decomp'

Parameters: double *inv

     lu: The LU decomposition of matrix A put into one matrix (ommit the
        diagonal of L).

     p: The list of permutations made during the pivoting of LU

     y: List with the ordenate values

Returns:

     x as list

EX:

    lu, p = lu.decomp(A)
    x = lu.solve(lu, p, y)
"
%enddef

%feature("docstring", SOLVE_LUDOC);
%rename(solve) solve_lu;
extern int solve_lu(double *lu, int dim, int *permut, double *y, double *x);
/* ************************************************************************** */


/* INV_LU */
/* ************************************************************************** */

/* The in typemap to recieve a 1D python list (vector) and convert ir to a C 1D
   array and allocate the memory for the output buffer vector. */
%typemap(in) (int *permut, double *inv)
{
    /* A tmp variable to hold the size of the PyList recieved */
    int tmpsize;

    /* Check and convert the input to a C array */
    $1 = PyList_to_int_vector($input, &tmpsize);

    /* Check if there was any error in the conversion */
    if(!$1)
    {
        return NULL;
    }

    /* Check if the size of the list is the same as the dimension of the system
       matrix A */
    if(tmpsize != lu_tmpdim)
    {
        /* Raise the appropriate exception */
        PyErr_SetString(PyExc_AttributeError, "Permutations list must have the same number of elements as A has lines.");
        free($1);
        return NULL;
    }
    
    /* Allocate the memory for the output buffer */
    $2 = (double *)malloc(tmpsize*tmpsize*sizeof(double));
}
/* This cleans up the array we malloc'd before the function call */
%typemap(freearg) (double *permut, double *inv)
{
    if($1)
    {
        free($1);
    }
}

/* Convert the temp output buffer to a Python list (the dimension will come
   in the result) */
%typemap(argout) (double *lu, int dim, int *permut, double *inv)
{
    /* First, check if there was any problem */
    if(!result)
    {
        /* Liberate the memory used for the output buffer */
        free($4);
        /* Raise the appropriate exception */
        PyErr_SetString(PyExc_MemoryError, "Memory allocation problem.");
        return NULL;
    }

    /* Blow away any previous result */
    Py_XDECREF($result);

    /* Convert the C vector to a PyList */
    $result = square_matrix_to_PyList($4, result);

    /* Liberate the memory used for the output buffer */
    free($4);
}


/* EXPOSE THE INV_LU FUNCTION */
%feature("autodoc", "1");
%define INV_LUDOC
"
Calculate the inverse of a square matrix given it's LU decomposition (with pivoting).
For LU decomposition, see function 'lu_decomp'

Parameters:

     lu: The LU decomposition of matrix A put into one matrix (ommit the
        diagonal of L).

     p: The list of permutations made during the pivoting of LU

Returns:

     inverse as a 2D list

EX:

    lu, p = lu.decomp(A)
    inverse = lu.inv(lu, p)
"
%enddef

%feature("docstring", INV_LUDOC);
%rename(inv) inv_lu;
extern int inv_lu(double *lu, int dim, int *permut, double *inv);
/* ************************************************************************** */
