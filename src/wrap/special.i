/* **************************************************************************
   Interface file for generation SWIG wrappers around 'special.c'

   OBS: Be careful with the typemaps! Using different types with the same
        argument names will cause wierd problems! 
   ************************************************************************** */
         
%define DOCSTRING
"
Special:

    Special function implementations such as the error function, the 
    complementary error function (and its derivative), etc.
    
Author: Leonardo Uieda
Created 14 Jul 2010
"
%enddef

%module(docstring=DOCSTRING) special


%{

#include "../c/special.h"

%}


%rename(erf) FAT_erf;
extern double FAT_erf(double x, double power_coef=1, double mult_coef=1, 
                      int ncells=100);
                      
           
%rename(erfc_deriv) FAT_erfc_deriv;           
extern double FAT_erfc_deriv(double x);