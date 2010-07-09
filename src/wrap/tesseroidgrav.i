/* **************************************************************************
   Interface file for generation SWIG wrappers around 'tesseroidgravity.c'

   OBS: Be careful with the typemaps! Using different types with the same
        argument names will cause wierd problems! 
   ************************************************************************** */
   
      
/* The module docstring */
%define DOCSTRING
"
meh    

Author: Leonardo Uieda
Created 17 Jun 2010
"
%enddef
/* Declare the module name */
%module(docstring=DOCSTRING) tesseroid


/* Put the headers with the definitions */
%{

#include "../c/tesseroidgrav.h"
#include "typeconversions.c"

%}

/* Typemaps to get the nodes and weights of the GLQ */

%typemap(in) (int glq_order, double *nodes)
{

    $2 = PyList_to_vector($input, &$1);
    
    /* Check if there was any error in the conversion */
    if(!$2)
    {
        return NULL;
    }
}

%typemap(freearg) (int glq_order, double *nodes)
{
    if($2)
    {
        free($2);
    }
}

%typemap(in) (double *weights)
{

    int tmp_order;

    $1 = PyList_to_vector($input, &tmp_order);
    
    /* Check if there was any error in the conversion */
    if(!$1)
    {
        return NULL;
    }
}

%typemap(freearg) (double *weights)
{
    if($1)
    {
        free($1);
    }
}


%feature("autodoc", "1");

%rename(gxx2d) tess_gxx2d;
extern double tess_gxx2d(double dens, double lon1, double lon2, double lat1,
       double lat2, double z1, double z2, double lonp, double latp,
       double zp, int glq_order, double *nodes, double *weights);
