/* **************************************************************************
   Interface file for generation SWIG wrappers around 'glq.c'

   OBS: Be careful with the typemaps! Using different types with the same
        argument names will cause wierd problems! 
   ************************************************************************** */
   
      
/* The module docstring */
%define DOCSTRING
"
QGL:
    C-coded set of functions for numerical integration via Gauss-Legendre 
    Quadrature (GLQ).    
    
    To integrate f(x) in the interval [a,b] using the GLQ:

       1) choose an order N for the quadrature (points in which f(x) will be
          discretized

       2) calculate the nodes using
           my_nodes = glq.nodes(N)

       3) calculate the weights using
           my_weights = glq.weights(my_nodes)

       4) scale the nodes to the interval [a,b] using
           my_scaled_nodes = glq.scale(a, b, my_nodes)

       5) do the summation of the weighted f(my_scaled_nodes)
           result = 0
           for i in range(N):
           
               result += my_weights[i]*f(my_scaled_nodes[i])
           
           # Account for the change in variables done when scaling the nodes
           result *= (b - a)/2
    

Author: Leonardo Uieda
Created 07 May 2010
"
%enddef
/* Declare the module name */
%module(docstring=DOCSTRING) glq


/* Put the headers with the definitions */
%{

#include "glq.h"
#include "typeconversions.c"

%}


/* GLQ_NODES */
/* ************************************************************************** */
/* Get the order of the GLQ and allocate memory for the output buffer */
%typemap(in) (int order, double *nodes)
{
    /* Set the order */
    $1 = PyInt_AsLong($input);
    
    /* malloc some memory for the output buffer */
    $2 = (double *)malloc($1*sizeof(double));
    
    /* Check if there was any error in the allocation */
    if(!$2)
    {
        return NULL;
    }
}

/* Convert the temp output buffer to a Python list (the order will come
   in the result) */
%typemap(argout) (int order, double *nodes)
{
    /* First, check if there was any problem */
    if(!result)
    {
        /* Liberate the memory used for the output buffer */
        free($2);
        /* Raise the appropriate exception */
        PyErr_SetString(PyExc_AttributeError, "Stagnation occured when calculating GLQ nodes.");
        return NULL;
    }

    /* Blow away any previous result */
    Py_XDECREF($result);

    /* Convert the C array to a Python List */
    $result = vector_to_PyList($2, result);

    /* Liberate the memory used for the output buffer */
    free($2);
}

/* EXPOSE THE GLQ_NODES FUNCTION */
%feature("autodoc", "1");
%define GLQ_NODES_DOC
"
Calculates the GLQ nodes. The nodes are the roots of the Legendre polynomial
of degree 'order'. The roots are calculated using Newton's Method for
Multiple Roots (Barrera-Figueroa et al., 2006).

Parameters:
    order: order of the GLQ

Returns:
    list with the nodes
"
%enddef

%feature("docstring", GLQ_NODES_DOC);
%rename(nodes) glq_nodes;
extern int glq_nodes(int order, double *nodes);

/* ************************************************************************** */


/* GLQ_WEIGHTS */
/* ************************************************************************** */
/* Get the order of the GLQ, the nodes and allocate memory for the output 
   buffer */
%typemap(in) (int order, double *nodes, double *)
{

    $2 = PyList_to_vector($input, &$1);
    
    /* Check if there was any error in the conversion */
    if(!$2)
    {
        return NULL;
    }
    
    /* malloc some memory for the output buffer */
    $3 = (double *)malloc($1*sizeof(double));
    
    /* Check if there was any error in the allocation */
    if(!$3)
    {
        return NULL;
    }
}
/* This cleans up the array we malloc'd before the function call */
%typemap(freearg) (int order, double *nodes, double *)
{
    if($2)
    {
        free($2);
    }
}

/* Convert the temp output buffer to a Python list (the order will come
   in the result) */
%typemap(argout) (int order, double *nodes, double *weights)
{
    /* First, check if there was any problem */
    if(!result)
    {
        /* Liberate the memory used for the output buffer */
        free($3);
        /* Raise the appropriate exception */
        PyErr_SetString(PyExc_ValueError, "Error calculating GLQ weights.");
        return NULL;
    }

    /* Blow away any previous result */
    Py_XDECREF($result);

    /* Convert the C array to a Python List */
    $result = vector_to_PyList($3, result);

    /* Liberate the memory used for the output buffer */
    free($3);
}

/* EXPOSE THE GLQ_WEIGHTS FUNCTION */
%feature("autodoc", "1");
%define GLQ_WEIGHTS_DOC
"
Calculates the weights for the GLQ.

Parameters:
    nodes: list with the GLQ nodes

Returns:
    list with the weights
"
%enddef

%feature("docstring", GLQ_WEIGHTS_DOC);
%rename(weights) glq_weights;
extern int glq_weights(int order, double *nodes, double *weights);
/* ************************************************************************** */



/* GLQ_SCALE_NODES */
/* ************************************************************************** */
/* NOTE: The in typemaps are the same as the ones for glq_weights */

/* Convert the temp output buffer to a Python list (the order will come
   in the result) */
%typemap(argout) (int order, double *nodes, double *scaled)
{
    /* First, check if there was any problem */
    if(!result)
    {
        /* Liberate the memory used for the output buffer */
        free($3);
        /* Raise the appropriate exception */
        PyErr_SetString(PyExc_ValueError, "Error scaling GLQ nodes.");
        return NULL;
    }

    /* Blow away any previous result */
    Py_XDECREF($result);

    /* Convert the C array to a Python List */
    $result = vector_to_PyList($3, result);

    /* Liberate the memory used for the output buffer */
    free($3);
}

/* EXPOSE THE GLQ_SCALE_NODES FUNCTION */
%feature("autodoc", "1");
%define GLQ_SCALE_NODES_DOC
"
Scales the GLQ nodes to the integration interval [a,b]. This is necessary
because the output of glq_nodes is scaled to [-1,1] (which is the value
gle_weights needs).

Parameters:
    a: lower integration limit
    b: upper integration limit
    nodes: list with the GLQ nodes

Returns:
    list with the scaled nodes
"
%enddef

%feature("docstring", GLQ_SCALE_NODES_DOC);
%rename(scale) glq_scale_nodes;
extern int glq_scale_nodes(double a, double b, int order, double *nodes,
                           double *scaled);
/* ************************************************************************** */
