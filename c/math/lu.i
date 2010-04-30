/* **************************************************************************
   Interface file for generation SWIG wrappers around 'lu.c'

   OBS: Be careful with the typemaps! Using different types with the same
        argument names will cause wierd problems! To try to solve that, I'll put
        a list of all typemaps here:
            lu_decomp_nopivot:
                typemap(in) (double *matrix, int dim, double *lu)
                typemap(freearg) (double *matrix, int dim, double *lu)
                typemap(argout) (double *matrix, int dim, double *lu)
            lu_decomp:
                typemap(in) (double *matrix, int dim, double *lu, int *permut)
                typemap(freearg) (double *matrix, int dim, double *lu, int *permut)
                typemap(argout) (double *matrix, int dim, double *lu, int *permut)
            solve_linsys_lu:
                typemap(in) (double *lu, int dim)
                typemap(in) (double *y, double *x)
                typemap(argout) (double *lu, int dim, double *y, double *x)
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

/* ************************************************************************** */


/* Put the headers with the definitions */
/* ************************************************************************** */
%{

#include "lu.h"

%}
/* ************************************************************************** */


/* TYPEMAPS */
/* ************************************************************************** */
/* ************************************************************************** */

/* HELPER FUNCTIONS */
/* ************************************************************************** */

/* PYLIST_TO_SQUARE_MATRIX */
%{
double *PyList_to_square_matrix(PyObject *input, int *dim)
{
    /* Check if the input is a list */
    if (!PyList_Check(input))
    {
        PyErr_SetString(PyExc_TypeError,"Input must be a list of equal sized lists (i.e. a 2D square matrix).");
        return NULL;
    }
    else
    {
        int tmpsize;
        int i, j;
        PyObject *obj, *item;
        double *result;

        /* Get the dimension of the matrix from the number of lines in the PyList */
        *dim = PyList_Size(input);

        /* Malloc some memory for the C matrix */
        result = (double *)malloc((*dim)*(*dim)*sizeof(double));

        /* Now check is each item in input is a list (making it a matrix)
           and convert them to a one dimensional C array matrix */
        for(i=0; i<*dim; i++)
        {
            obj = PyList_GetItem(input, i);

            /* If it is not a list, cast an exception telling so */
            if(!PyList_Check(obj))
            {
                PyErr_SetString(PyExc_TypeError,"List passed must contain lists (i.e. be a 2D matrix).");
                free(result);
                return NULL;
            }

            /* Get the size of this line and see if it matches the dim */
            tmpsize = PyList_Size(obj);
            if(tmpsize != *dim)
            {
                PyErr_SetString(PyExc_AttributeError, "Must be a square matrix.");
                free(result);
                return NULL;
            }

            /* Now get the items out of this list */
            for(j=0; j<*dim; j++)
            {
                item = PyList_GetItem(obj, j);

                /* Check is the item is a double */
                if(!PyNumber_Check(item))
                {
                    PyErr_SetString(PyExc_TypeError,"Must be 2D array of floats.");
                    free(result);
                    return NULL;
                }

                /* Put it in the right position in the C matrix */
                result[POS(i, j, *dim)] = PyFloat_AsDouble(item);
            }
        }

        /* Returned the allocated C array with the matrix */
        return result;
    }
}
%}

/* SQUARE_MATRIX_TO_PYLIST */
%{
PyObject *square_matrix_to_PyList(double *matrix, int dim)
{
    int i, j;
    PyObject *tmp, *result;

    /* Create the new list object */
    result = PyList_New(dim);

    /* Now set the elements in it with the elements of matrix */
    for(i=0; i<dim; i++)
    {
        /* Create a new buffer row */
        tmp = PyList_New(dim);
        for(j=0; j<dim; j++)
        {
            PyList_SetItem(tmp, j, PyFloat_FromDouble(matrix[POS(i,j,dim)]));
        }

        /* Now put the row in the $result */
        PyList_SetItem(result, i, tmp);
    }

    /* Return the Python List */
    return result;
}
%}

/* PYLIST_TO_VECTOR */
%{
double *PyList_to_vector(PyObject *input, int *size)
{
    /* Check if the input is a list */
    if (!PyList_Check(input))
    {
        PyErr_SetString(PyExc_TypeError,"Input must be a list.");
        return NULL;
    }
    else
    {
        int i;
        PyObject *item;
        double *result;

        /* Get the size of the vector from the number of lines in the PyList */
        *size = PyList_Size(input);

        /* Malloc some memory for the c vector */
        result = (double *)malloc((*size)*sizeof(double));

        /* Check if each item in input is a float and put them in the array */
        for(i=0; i<*size; i++)
        {
            item = PyList_GetItem(input, i);

            /* Check is the item is a float or int */
            if(!PyNumber_Check(item))
            {
                PyErr_SetString(PyExc_TypeError,"Input must be a list of floats or ints.");
                free(result);
                return NULL;
            }

            /* Put it in the right position in the C array */
            result[i] = PyFloat_AsDouble(item);
        }

        /* Return the C array */
        return result;
    }
}
%}

/* PYLIST_TO_INT_VECTOR */
%{
int *PyList_to_int_vector(PyObject *input, int *size)
{
    /* Check if the input is a list */
    if (!PyList_Check(input))
    {
        PyErr_SetString(PyExc_TypeError,"Input must be a list.");
        return NULL;
    }
    else
    {
        int i;
        PyObject *item;
        int *result;

        /* Get the size of the vector from the number of lines in the PyList */
        *size = PyList_Size(input);

        /* Malloc some memory for the c vector */
        result = (int *)malloc((*size)*sizeof(int));

        /* Check if each item in input is an int and put them in the array */
        for(i=0; i<*size; i++)
        {
            item = PyList_GetItem(input, i);

            /* Check is the item is an int */
            if(!PyInt_Check(item))
            {
                PyErr_SetString(PyExc_TypeError,"Input must be a list of ints.");
                free(result);
                return NULL;
            }

            /* Put it in the right position in the C array */
            result[i] = PyInt_AsLong(item);
        }

        /* Return the C array */
        return result;
    }
}
%}

/* INT_VECTOR_TO_PYLIST */
%{
PyObject *int_vector_to_PyList(int *vec, int size)
{
    int i;
    PyObject *result;

    /* Create a list to put the vector in */
    result = PyList_New(size);

    /* Now set the elements in it with the elements of the vector */
    for(i=0; i<size; i++)
    {
        PyList_SetItem(result, i, PyInt_FromLong(vec[i]));
    }

    /* Return the Python List */
    return result;
}
%}

/* VECTOR_TO_PYLIST */
%{
PyObject *vector_to_PyList(double *vec, int size)
{
    int i;
    PyObject *result;

    /* Create a list to put the vector in */
    result = PyList_New(size);

    /* Now set the elements in it with the elements of the vector */
    for(i=0; i<size; i++)
    {
        PyList_SetItem(result, i, PyFloat_FromDouble(vec[i]));
    }

    /* Return the Python List */
    return result;
}
%}
/* ************************************************************************** */


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


/* SOLVE_LINSYS_LU */
/* ************************************************************************** */

/* A temp variable used to check if the dimension of the input matrix A is the
   same as the input vector y */
%{
int lu_decomp_pivot_tmpdim;
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
    lu_decomp_pivot_tmpdim = $2;
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
    if(tmpsize != lu_decomp_pivot_tmpdim)
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
   array. Also allocated the memory for the output buffer vector. */
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
    if(tmpsize != lu_decomp_pivot_tmpdim)
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

/* EXPOSE THE SOLVE_LINSYS_LU FUNCTION */
%feature("autodoc", "1");
%define SOLVE_LINSYS_LUDOC
"
Solve a linear system Ax=y given it's LU decomposition (with pivoting).
For LU decomposition, see function 'lu_decomp'

Parameters:

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

%feature("docstring", SOLVE_LINSYS_LUDOC);
%rename(solve) solve_lu;
extern int solve_lu(double *lu, int dim, int *permut, double *y, double *x);
/* ************************************************************************** */

/* ************************************************************************** */
/* ************************************************************************** */
