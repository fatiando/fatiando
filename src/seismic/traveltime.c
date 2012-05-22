/* **************************************************************************

 Functions that calculate the travel times of seismic waves.

 **************************************************************************** */

#include <math.h>
//#include <stdio.h>


/* Calculate the travel time inside a 2D square cell assuming the ray is a
   straight line.

 Parameters:
 * 
    * double velocity: the velocity of the cell. Must be in units compatible
        with the other parameters!
    * double x1, y1: coordinates of the lower-left corner of the cell
    * double x2, y2: coordinates of the upper-right corner of the cell
    * double *x_src, *y_src: coordinates of the wave sources
    * double *x_rec, *y_rec: coordinates of the receivers
    * unsigned int n: number of source-receiver pairs
    * double *times: array to return the computed times for each source-receiver
                     pair
*/
unsigned int straight_ray_2d(double velocity, double x1, double y1, double x2,
    double y2, double *x_src, double *y_src, double *x_rec, double *y_rec,
    unsigned int n, double *times)
{
    double maxx, maxy, minx, miny, distance;
    double xps[6], yps[6], xp, yp;
    double crossingx[6], crossingy[6];
    double a_ray, b_ray;
    int crossingsize, inside;
    register unsigned int i, j, l;
    short duplicate;

    for(l=0; l < n; l++, times++, x_src++, y_src++, x_rec++, y_rec++)
    {
        /* Some aux variables to avoid calling max and min too much */
        #define MAX(a,b) ((a) > (b) ? (a) : (b))
        #define MIN(a,b) ((a) < (b) ? (a) : (b))
        maxx = MAX(*x_src, *x_rec);
        maxy = MAX(*y_src, *y_rec);
        minx = MIN(*x_src, *x_rec);
        miny = MIN(*y_src, *y_rec);
        #undef MAX
        #undef MIN
        /* Check if the cell is in the rectangle with the ray path as a
         * diagonal. If not, then the ray doesn't go through the cell. */
        if(x2 < minx || x1 > maxx || y2 < miny || y1 > maxy)
        {
            *times = 0;
            continue;
        }
        /* Vertical case */
        if((*x_rec - *x_src) == 0)
        {
            /* Find the places where the ray intersects the cell */
            xps[0] = *x_rec;
            xps[1] = *x_rec;
            xps[2] = *x_rec;
            xps[3] = *x_rec;
            yps[0] = *y_rec;
            yps[1] = *y_src;
            yps[2] = y1;
            yps[3] = y2;
            crossingsize = 4;
        }
        /* Horizontal case */
        else if((*y_rec - *y_src) == 0)
        {
            /* Find the places where the ray intersects the cell */
            xps[0] = *x_rec;
            xps[1] = *x_src;
            xps[2] = x1;
            xps[3] = x2;
            yps[0] = *y_rec;
            yps[1] = *y_rec;
            yps[2] = *y_rec;
            yps[3] = *y_rec;
            crossingsize = 4;
        }
        else
        {
            a_ray = (double)(*y_rec - *y_src)/(*x_rec - *x_src);
            b_ray = *y_src - a_ray*(*x_src);
            /* Find the places where the ray intersects the cell */
            xps[0] = x1;
            xps[1] = x2;
            yps[0] = a_ray*x1 + b_ray;
            yps[1] = a_ray*x2 + b_ray;
            yps[2] = y1;
            yps[3] = y2;
            xps[2] = (double)(y1 - b_ray)/a_ray;
            xps[3] = (double)(y2 - b_ray)/a_ray;
            /* Add the src and rec locations so that the travel time of a src
             * or rec inside a cell is accounted for */
            xps[4] = *x_src;
            xps[5] = *x_rec;
            yps[4] = *y_src;
            yps[5] = *y_rec;
            crossingsize = 6;
        }
        /* Find out how many points are inside both the cell and the rectangle
         * with the ray path as a diagonal */
        inside = 0;
        for(i=0; i < crossingsize; i++)
        {
            xp = xps[i];
            yp = yps[i];
            if( (xp <= x2 && xp >= x1 && yp <= y2 && yp >= y1) &&
                (xp <= maxx && xp >= minx && yp <= maxy && yp >= miny))
            {
                duplicate = 0;
                for(j=0; j < inside; j++)
                {
                    if(crossingx[j] == xp && crossingy[j] == yp)
                    {
                        duplicate = 1;
                        break;
                    }
                }
                if(!duplicate)
                {
                    crossingx[inside] = xp;
                    crossingy[inside] = yp;
                    inside++;
                }
            }
        }
        //if(inside > 2)
        //{
            //fprintf(stderr, "Error calculating trave ltime in straight_ray_2d");
        //}
        if(inside < 2)
        {
            *times = 0;
        }
        else
        {
            distance = sqrt((crossingx[1] - crossingx[0])*
                            (crossingx[1] - crossingx[0]) +
                            (crossingy[1] - crossingy[0])*
                            (crossingy[1] - crossingy[0]));
            *times = (double)distance/velocity;
        }
    }
    return l;
}
