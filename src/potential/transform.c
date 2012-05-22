/* **************************************************************************

   Potential field transformations.

   ************************************************************************** */

#include <math.h>

const double PI = 3.1415926535897932384626433832795;

/* Perform the upward continuation of gravity data using the analytical formula

Data must be on a regular grid!

Parameters:

    * double *gz: the original gravity data (observations)
    * double z0: original z coordinate of the observations
                 (remember that z->down)
    * double newz: new z coordinate of the continued observations
    * double *xp, *yp: x and y coordinates of the observation points
    * double dx, dy: grid spacing in the x and y dimensions
    * unsigned int n: number of observation points
    * double *gzcont: vector used to return the upward continued gz on the n
                      points
    
Returns:

    * unsigned int: number of points calculated
*/


unsigned int upcontinue(double *gz, double z0, double newz, double *xp,
    double *yp, double dx, double dy, unsigned int n, double *gzcont)
{
    register unsigned int i, j;
    double *x, *y, *xl, *yl, *g, area, oneover_l, deltaz_sqr, fact;

    area = dx*dy;
    x = xp;
    y = yp;
    deltaz_sqr = (z0 - newz)*(z0 - newz);
    #define FAT_ABS(x) ((x) < 0 ? -1*(x) : (x))
    fact = FAT_ABS(z0 - newz)/(2*PI);
    #undef FAT_ABS
    for(i=0; i < n; i++)
    {
        xl = xp;
        yl = yp;
        *gzcont = 0;
        g = gz;
        for(j=0; j < n; j++)
        {
            oneover_l = pow((*x - *xl)*(*x - *xl) + (*y - *yl)*(*y - *yl) +
                            deltaz_sqr, -1.5);
            *gzcont += (*g)*oneover_l*area;
            xl++;
            yl++;
            g++;
        }
        *gzcont *= fact;
        gzcont++;
        x++;
        y++;
    }
    return i;
}
