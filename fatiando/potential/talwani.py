"""
Calculate the gravitational attraction of a 2D body with polygonal vertical
cross-section using the formula of Talwani et al. (1959)

Use the :func:`~fatiando.mesher.dd.Polygon` object to create polygons.

.. warning:: the vertices must be given clockwise! If not, the result will have
    an inverted sign.

**Components**

* :func:`~fatiando.potential.talwani.gz`

**References**

Talwani, M., J. L. Worzel, and M. Landisman (1959), Rapid Gravity Computations
for Two-Dimensional Bodies with Application to the Mendocino Submarine
Fracture Zone, J. Geophys. Res., 64(1), 49-59, doi:10.1029/JZ064i001p00049.

----

"""
import numpy

from fatiando import logger

log = logger.dummy('fatiando.potential.talwani')

# The gravitational constant (m^3*kg^-1*s^-1)
G = 0.00000000006673
# Conversion factor from SI units to mGal: 1 m/s**2 = 10**5 mGal
SI2MGAL = 100000.0


def gz(xp, zp, polygons):
    """
    Calculates the :math:`g_z` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:
    
    * xp, zp : arrays
        The x and z coordinates of the computation points.        
    * polygons : list of :func:`~fatiando.mesher.dd.Polygon`
        The density model used.
        Polygons must have the property ``'density'``. Polygons that don't have
        this property will be ignored in the computations. Elements of
        *polygons* that are None will also be ignored.

        .. note:: The y coordinate of the polygons is used as z! 

    Returns:
    
    * gz : array
        The :math:`g_z` component calculated on the computation points

    """
    if xp.shape != zp.shape:
        raise ValueError("Input arrays xp and zp must have same shape!")
    res = numpy.zeros_like(xp)
    for p in polygons:
        if p is None or 'density' in p:
            

            
            res += _talwani.talwani_gz(float(p['density']), p['x'], p['y'],
                xp, zp)
    return res



#unsigned int talwani_gz(double dens, double *x, double *z, unsigned int m,
                        #double *xp, double *zp, unsigned int n, double *res)
#{
    #double *px, *pz;
    #double xv, zv, xvp1, zvp1, theta_v, theta_vp1, phi_v, ai, tmp;
    #int flag;
    #register unsigned int i, v;
    #
    #for(i=0; i < n; i++, res++, xp++, zp++)
    #{
        #flag = 0;
        #*res = 0;
        #tmp = 0;
        #xvp1 = *x - *xp;
        #zvp1 = *z - *zp;  
        #px = x;
        #pz = z;  
        #for(v=0; v < m; v++)
        #{
            #xv = xvp1;
            #zv = zvp1;
            #/* The last vertice pairs with the first one */
            #if(v == m - 1)
            #{
                #xvp1 = *x - *xp;
                #zvp1 = *z - *zp;
            #}
            #else
            #{
                #xvp1 = *(++px) - *xp;
                #zvp1 = *(++pz) - *zp;                
            #}
            #/* Temporary fix to the two bad conditions bellow */
            #if(xv == 0 || xv == xvp1)
            #{
                #xv += 0.1;
            #}
            #if((xv == 0. && zv == 0.) || zv == zvp1)
            #{
                #zv += 0.1;
            #}
            #if(xvp1 == 0. && zvp1 == 0.)
            #{
                #zvp1 += 0.1;
            #}
            #if(xvp1 == 0.)
            #{
                #xvp1 += 0.1;
            #}
            #/* Fix ends here */
            #theta_v = atan2(zv, xv); 
            #theta_vp1 = atan2(zvp1, xvp1); 
            #phi_v = atan2(zvp1 - zv, xvp1 - xv); 
            #ai = xvp1 + (zvp1)*((double)(xvp1 - xv)/(zv - zvp1));            
            #if(theta_v < 0)
            #{
                #theta_v += FAT_PI;
            #}
            #if(theta_vp1 < 0)
            #{
                #theta_vp1 += FAT_PI;
            #}   
            #/* There is something wrong with these conditions. Need to review.
             #* For now, just sum 0.1 meter to one of the coordinates (above).
             #* Gives decent enough result.          
            #if(xv == 0)
            #{
                #tmp = -ai*sin(phi_v)*cos(phi_v)*(theta_vp1 -
                    #0.5*FAT_PI + tan(phi_v)*log(
                        #cos(theta_vp1)*(tan(theta_vp1)- tan(phi_v))));
                #flag = 1;
            #}             
            #if(xvp1 == 0)
            #{
                #tmp = ai*sin(phi_v)*cos(phi_v)*(theta_v -
                    #0.5*FAT_PI + tan(phi_v)*log(
                        #cos(theta_v)*(tan(theta_v) - tan(phi_v))));         
                #flag = 1;
            #}
            #if(zv == zvp1)
            #{
                #tmp = zv*(theta_vp1 - theta_v);
                #flag = 1; 
            #}            
            #if(xv == xvp1)
            #{
                #tmp = xv*(log((double)cos(theta_v)/cos(theta_vp1)));
                #flag = 1;
            #}         
            #if((theta_v == theta_vp1) || (xv == 0. && zv == 0.) ||
               #(xvp1 == 0. && zvp1 == 0.))
            #{
                #tmp = 0;
                #flag = 1;
            #}*/
            #if(theta_v == theta_vp1)
            #{
                #tmp = 0;
                #flag = 1;
            #}
            #if(!flag)
            #{ 
                #tmp = ai*sin(phi_v)*cos(phi_v)*(theta_v - theta_vp1 +
                    #tan(phi_v)*log((double)
                        #(cos(theta_v)*(tan(theta_v) - tan(phi_v)))/
                        #(cos(theta_vp1)*(tan(theta_vp1) - tan(phi_v)))));
            #}
            #*res += tmp;
        #}
        #*res *= SI2MGAL*2.0*G*dens;
    #}
    #return i;
#}
