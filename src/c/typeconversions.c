/* **************************************************************************

   Some helper functions to convert between Python lists and C arrays.

   Functions:

    double *PyList_to_square_matrix(PyObject *input, int *dim)
    PyObject *square_matrix_to_PyList(double *matrix, int dim)

    double *PyList_to_vector(PyObject *input, int *size)
    PyObject *vector_to_PyList(double *vec, int size)

    int *PyList_to_int_vector(PyObject *input, int *size)
    PyObject *int_vector_to_PyList(int *vec, int size)

   ************************************************************************** */

/* Macro for accessing the element i,j of a matrix with ncols columns. This
matrix should be laid out in a 1D array. */
#ifndef POS
#define POS(i,j,ncols) (((i)*(ncols)) + (j))
#endif


/* PYLIST_TO_SQUARE_MATRIX */
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


/* SQUARE_MATRIX_TO_PYLIST */
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


/* PYLIST_TO_VECTOR */
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


/* PYLIST_TO_INT_VECTOR */
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


/* INT_VECTOR_TO_PYLIST */
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


/* VECTOR_TO_PYLIST */
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
