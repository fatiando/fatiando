#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* CONSTANTS */
/* ************************************************************************** */

/* The gravitational constant (km^3*kg^-1*s^-2) */
#define GRAV 6.67E-20

/* Constante real positiva que evita singularidades numéricas */
#define EVITA_SINGULARIDADE_NUMERICA 1E-10

double cilin_pol_grav_tensor (int N, int M, int *nvertices, double *Xp, double *Yp,
							double *Zp, double **tensor, double *rho, double *z1,
							double *z2, double **x, double **y) {

	int i, j, k;
	double X1, X2, Y1, Y2, Z1, Z2;
	double V[6];
	double aux;

	/* Cálculo do tensor gravitacional ==> */

	for (i = 0; i < N; i++) { /* looping dos pontos */

		for (j = 0; j < M; j++) { /* looping dos prismas */

			printf ("i = %d / %d j = %d / %d\n", i, N, j, M);

			/*aux = GRAV*rho[j]*CONVERT2EOTVOS*;*/
			//aux = GRAV*rho[j]*1E12*1E9;
			aux = 1.0;
			/* o 1E12 transforma a densidade de g/cm^3 para kg/km^3 */
			/* o 1E9 transforma para Eotvos */

			Z1 = z1[j] - Zp[i];
			Z2 = z2[j] - Zp[i];

			Z1 *= 0.001;
			Z2 *= 0.001;

			V[0] = 0.0;
			V[1] = 0.0;
			V[2] = 0.0;
			V[3] = 0.0;
			V[4] = 0.0;
			V[5] = 0.0;

			for (k = 0; k < (nvertices[j] - 1); k++) { /* looping dos vértices */

				X1 = x[j][k] - Xp[i];
				Y1 = y[j][k] - Yp[i];
				X2 = x[j][k+1] - Xp[i];
				Y2 = y[j][k+1] - Yp[i];

				X1 *= 0.001;
				X2 *= 0.001;
				Y1 *= 0.001;
				Y2 *= 0.001;

				V[0] += integral_aresta_gxx (X1, X2, Y1, Y2, Z1, Z2);
				V[1] += integral_aresta_gxy (X1, X2, Y1, Y2, Z1, Z2);
				V[2] += integral_aresta_gxz (X1, X2, Y1, Y2, Z1, Z2);
				V[3] += integral_aresta_gyy (X1, X2, Y1, Y2, Z1, Z2);
				V[4] += integral_aresta_gyz (X1, X2, Y1, Y2, Z1, Z2);
				V[5] += integral_aresta_gzz (X1, X2, Y1, Y2, Z1, Z2);

			} /* loooping dos vértices */

			/* cálculo referente a última aresta ==> */

				X1 = x[j][k] - Xp[i];
				Y1 = y[j][k] - Yp[i];
				X2 = x[j][0] - Xp[i];
				Y2 = y[j][0] - Yp[i];

				X1 *= 0.001;
				X2 *= 0.001;
				Y1 *= 0.001;
				Y2 *= 0.001;

				V[0] += integral_aresta_gxx (X1, X2, Y1, Y2, Z1, Z2);
				V[1] += integral_aresta_gxy (X1, X2, Y1, Y2, Z1, Z2);
				V[2] += integral_aresta_gxz (X1, X2, Y1, Y2, Z1, Z2);
				V[3] += integral_aresta_gyy (X1, X2, Y1, Y2, Z1, Z2);
				V[4] += integral_aresta_gyz (X1, X2, Y1, Y2, Z1, Z2);
				V[5] += integral_aresta_gzz (X1, X2, Y1, Y2, Z1, Z2);

			/* <== cálculo referente a última aresta */

			tensor[0][i] += aux*V[0]; /* componente xx */
			tensor[1][i] += aux*V[1]; /* componente xy */
			tensor[2][i] += aux*V[2]; /* componente xz */
			tensor[3][i] += aux*V[3]; /* componente yy */
			tensor[4][i] += aux*V[4]; /* componente yz */
			tensor[5][i] += aux*V[5]; /* componente zz */

		} /* looping dos prismas */

	} /* looping dos pontos */

	/* <== Cálculo do tensor gravitacional */

	return 0;

}

double integral_aresta_gxx (double X1, double X2, double Y1, double Y2, double Z1, double Z2) {

	double n, p, m, c, g, R11, R12, R21, R22, d1, d2;
	double aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
	double aux9, aux10, aux11, aux12, aux13, aux14, aux15, aux16;
	double gxxaux;

	gxxaux = 0.0;

	aux0 = X2 - X1 + EVITA_SINGULARIDADE_NUMERICA;
	aux1 = Y2 - Y1 + EVITA_SINGULARIDADE_NUMERICA;

	n = (double)(aux0/aux1);
	g = X1 - (Y1*n);

	m = (double)(aux1/aux0);
	c = Y1 - (X1*m);

	aux2 = pow(((aux0*aux0) + (aux1*aux1)), 0.5);
	aux3 = (X1*Y2) - (X2*Y1);

	p = ((double)(aux3/aux2)) + EVITA_SINGULARIDADE_NUMERICA;

	aux4 = (aux0*X1) + (aux1*Y1);
	aux5 = (aux0*X2) + (aux1*Y2);

	d1 = ((double)(aux4/aux2)) + EVITA_SINGULARIDADE_NUMERICA;
	d2 = ((double)(aux5/aux2)) + EVITA_SINGULARIDADE_NUMERICA;

	aux6 = (X1*X1) + (Y1*Y1);
	aux7 = (X2*X2) + (Y2*Y2);
	aux8 = Z1*Z1;
	aux9 = Z2*Z2;

	R11 = pow((aux6 + aux8), 0.5);
	R12 = pow((aux6 + aux9), 0.5);
	R21 = pow((aux7 + aux8), 0.5);
	R22 = pow((aux7 + aux9), 0.5);

	aux10 = atan((double)((Z2*d2)/(p*R22)));
	aux11 = atan((double)((Z1*d2)/(p*R21)));
	aux12 = aux10 - aux11;
	aux13 = (double)(aux12/(p*d2));
	aux14 = (double)((p*aux12)/d2);

	gxxaux += (g*Y2*aux13) + (n*aux14);

	aux10 = atan((double)((Z2*d1)/(p*R12)));
	aux11 = atan((double)((Z1*d1)/(p*R11)));
	aux12 = aux10 - aux11;
	aux13 = (double)(aux12/(p*d1));
	aux14 = (double)((p*aux12)/d1);

	gxxaux -= (g*Y1*aux13) + (n*aux14);

	aux10 = log(((Z2 + R22) + EVITA_SINGULARIDADE_NUMERICA));
	aux11 = log(((Z1 + R21) + EVITA_SINGULARIDADE_NUMERICA));
	aux12 = log(((Z2 + R12) + EVITA_SINGULARIDADE_NUMERICA));
	aux13 = log(((Z1 + R11) + EVITA_SINGULARIDADE_NUMERICA));
	aux14 = aux10 - aux11;
	aux15 = aux12 - aux13;

	gxxaux += (n*(aux15 - aux14));

	aux0 = (double)(1.0/(1.0 + (n*n)));

	gxxaux *= -aux0;

	return gxxaux;

}

double integral_aresta_gxy (double X1, double X2, double Y1, double Y2, double Z1, double Z2) {

	double n, p, m, c, g, R11, R12, R21, R22, d1, d2;
	double aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
	double aux9, aux10, aux11, aux12, aux13, aux14, aux15, aux16;
	double gxyaux;

	gxyaux = 0.0;

	aux0 = X2 - X1 + EVITA_SINGULARIDADE_NUMERICA;
	aux1 = Y2 - Y1 + EVITA_SINGULARIDADE_NUMERICA;

	n = (double)(aux0/aux1);
	g = X1 - (Y1*n);

	m = (double)(aux1/aux0);
	c = Y1 - (X1*m);

	aux2 = pow(((aux0*aux0) + (aux1*aux1)), 0.5);
	aux3 = (X1*Y2) - (X2*Y1);

	p = ((double)(aux3/aux2)) + EVITA_SINGULARIDADE_NUMERICA;

	aux4 = (aux0*X1) + (aux1*Y1);
	aux5 = (aux0*X2) + (aux1*Y2);

	d1 = ((double)(aux4/aux2)) + EVITA_SINGULARIDADE_NUMERICA;
	d2 = ((double)(aux5/aux2)) + EVITA_SINGULARIDADE_NUMERICA;

	aux6 = (X1*X1) + (Y1*Y1);
	aux7 = (X2*X2) + (Y2*Y2);
	aux8 = Z1*Z1;
	aux9 = Z2*Z2;

	R11 = pow((aux6 + aux8), 0.5);
	R12 = pow((aux6 + aux9), 0.5);
	R21 = pow((aux7 + aux8), 0.5);
	R22 = pow((aux7 + aux9), 0.5);

	aux10 = atan((double)((Z2*d2)/(p*R22)));
	aux11 = atan((double)((Z1*d2)/(p*R21)));
	aux12 = aux10 - aux11;
	aux13 = (double)(aux12/(p*d2));
	aux14 = (double)((p*aux12)/d2);

	gxyaux += (((g*g) + (g*n*Y2))*aux13) - aux14;

	aux10 = atan((double)((Z2*d1)/(p*R12)));
	aux11 = atan((double)((Z1*d1)/(p*R11)));
	aux12 = aux10 - aux11;
	aux13 = (double)(aux12/(p*d1));
	aux14 = (double)((p*aux12)/d1);

	gxyaux -= (((g*g) + (g*n*Y1))*aux13) - aux14;

	aux10 = log(((Z2 + R22) + EVITA_SINGULARIDADE_NUMERICA));
	aux11 = log(((Z1 + R21) + EVITA_SINGULARIDADE_NUMERICA));
	aux12 = log(((Z2 + R12) + EVITA_SINGULARIDADE_NUMERICA));
	aux13 = log(((Z1 + R11) + EVITA_SINGULARIDADE_NUMERICA));
	aux14 = aux10 - aux11;
	aux15 = aux12 - aux13;

	gxyaux += (aux14 - aux15);

	aux0 = (double)(1.0/(1.0 + (n*n)));

	gxyaux *= aux0;

	return gxyaux;

}

double integral_aresta_gxz (double X1, double X2, double Y1, double Y2, double Z1, double Z2) {

	double n, p, m, c, g, R11, R12, R21, R22, d1, d2;
	double aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
	double aux9, aux10, aux11, aux12, aux13, aux14, aux15, aux16;
	double gxzaux;

	gxzaux = 0.0;

	aux0 = X2 - X1 + EVITA_SINGULARIDADE_NUMERICA;
	aux1 = Y2 - Y1 + EVITA_SINGULARIDADE_NUMERICA;

	n = (double)(aux0/aux1);
	g = X1 - (Y1*n);

	m = (double)(aux1/aux0);
	c = Y1 - (X1*m);

	aux2 = pow(((aux0*aux0) + (aux1*aux1)), 0.5);
	aux3 = (X1*Y2) - (X2*Y1);

	p = ((double)(aux3/aux2)) + EVITA_SINGULARIDADE_NUMERICA;

	aux4 = (aux0*X1) + (aux1*Y1);
	aux5 = (aux0*X2) + (aux1*Y2);

	d1 = ((double)(aux4/aux2)) + EVITA_SINGULARIDADE_NUMERICA;
	d2 = ((double)(aux5/aux2)) + EVITA_SINGULARIDADE_NUMERICA;

	aux6 = (X1*X1) + (Y1*Y1);
	aux7 = (X2*X2) + (Y2*Y2);
	aux8 = Z1*Z1;
	aux9 = Z2*Z2;

	R11 = pow((aux6 + aux8), 0.5);
	R12 = pow((aux6 + aux9), 0.5);
	R21 = pow((aux7 + aux8), 0.5);
	R22 = pow((aux7 + aux9), 0.5);

	aux10 = log((((R11 - d1)/(R11 + d1)) + EVITA_SINGULARIDADE_NUMERICA));
	aux11 = log((((R12 - d1)/(R12 + d1)) + EVITA_SINGULARIDADE_NUMERICA));
	aux12 = log((((R21 - d2)/(R21 + d2)) + EVITA_SINGULARIDADE_NUMERICA));
	aux13 = log((((R22 - d2)/(R22 + d2)) + EVITA_SINGULARIDADE_NUMERICA));
	aux14 = (double)(1.0/(2*d1));
	aux15 = (double)(1.0/(2*d2));
	aux16 = aux15*(aux13 - aux12);

	gxzaux += (Y2*(1.0 + (n*n)) + g*n)*aux16;

	aux16 = aux14*(aux11 - aux10);

	gxzaux -= (Y1*(1.0 + (n*n)) + g*n)*aux16;

	aux0 = (double)(1.0/(1.0 + (n*n)));

	gxzaux *= -aux0;

	return gxzaux;

}

double integral_aresta_gyy (double X1, double X2, double Y1, double Y2, double Z1, double Z2) {

	double n, p, m, c, g, R11, R12, R21, R22, d1, d2;
	double aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
	double aux9, aux10, aux11, aux12, aux13, aux14, aux15, aux16;
	double gyyaux;

	gyyaux = 0.0;

	aux0 = X2 - X1 + EVITA_SINGULARIDADE_NUMERICA;
	aux1 = Y2 - Y1 + EVITA_SINGULARIDADE_NUMERICA;

	n = (double)(aux0/aux1);
	g = X1 - (Y1*n);

	m = (double)(aux1/aux0);
	c = Y1 - (X1*m);

	aux2 = pow(((aux0*aux0) + (aux1*aux1)), 0.5);
	aux3 = (X1*Y2) - (X2*Y1);

	p = ((double)(aux3/aux2)) + EVITA_SINGULARIDADE_NUMERICA;

	aux4 = (aux0*X1) + (aux1*Y1);
	aux5 = (aux0*X2) + (aux1*Y2);

	d1 = ((double)(aux4/aux2)) + EVITA_SINGULARIDADE_NUMERICA;
	d2 = ((double)(aux5/aux2)) + EVITA_SINGULARIDADE_NUMERICA;

	aux6 = (X1*X1) + (Y1*Y1);
	aux7 = (X2*X2) + (Y2*Y2);
	aux8 = Z1*Z1;
	aux9 = Z2*Z2;

	R11 = pow((aux6 + aux8), 0.5);
	R12 = pow((aux6 + aux9), 0.5);
	R21 = pow((aux7 + aux8), 0.5);
	R22 = pow((aux7 + aux9), 0.5);

	aux10 = atan((double)((Z2*d2)/(p*R22)));
	aux11 = atan((double)((Z1*d2)/(p*R21)));
	aux12 = aux10 - aux11;
	aux13 = (double)(aux12/(p*d2));
	aux14 = (double)((p*aux12)/d2);

	gyyaux += (c*X2*aux13) + (m*aux14);

	aux10 = atan((double)((Z2*d1)/(p*R12)));
	aux11 = atan((double)((Z1*d1)/(p*R11)));
	aux12 = aux10 - aux11;
	aux13 = (double)(aux12/(p*d1));
	aux14 = (double)((p*aux12)/d1);

	gyyaux -= (c*X1*aux13) + (m*aux14);

	aux10 = log(((Z2 + R22) + EVITA_SINGULARIDADE_NUMERICA));
	aux11 = log(((Z1 + R21) + EVITA_SINGULARIDADE_NUMERICA));
	aux12 = log(((Z2 + R12) + EVITA_SINGULARIDADE_NUMERICA));
	aux13 = log(((Z1 + R11) + EVITA_SINGULARIDADE_NUMERICA));
	aux14 = aux10 - aux11;
	aux15 = aux12 - aux13;

	gyyaux += (m*(aux15 - aux14));

	aux1 = (double)(1.0/(1.0 + (m*m)));

	gyyaux *= aux1;

	return gyyaux;

}

double integral_aresta_gyz (double X1, double X2, double Y1, double Y2, double Z1, double Z2) {

	double n, p, m, c, g, R11, R12, R21, R22, d1, d2;
	double aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
	double aux9, aux10, aux11, aux12, aux13, aux14, aux15, aux16;
	double gyzaux;

	gyzaux = 0.0;

	aux0 = X2 - X1 + EVITA_SINGULARIDADE_NUMERICA;
	aux1 = Y2 - Y1 + EVITA_SINGULARIDADE_NUMERICA;

	n = (double)(aux0/aux1);
	g = X1 - (Y1*n);

	m = (double)(aux1/aux0);
	c = Y1 - (X1*m);

	aux2 = pow(((aux0*aux0) + (aux1*aux1)), 0.5);
	aux3 = (X1*Y2) - (X2*Y1);

	p = ((double)(aux3/aux2)) + EVITA_SINGULARIDADE_NUMERICA;

	aux4 = (aux0*X1) + (aux1*Y1);
	aux5 = (aux0*X2) + (aux1*Y2);

	d1 = ((double)(aux4/aux2)) + EVITA_SINGULARIDADE_NUMERICA;
	d2 = ((double)(aux5/aux2)) + EVITA_SINGULARIDADE_NUMERICA;

	aux6 = (X1*X1) + (Y1*Y1);
	aux7 = (X2*X2) + (Y2*Y2);
	aux8 = Z1*Z1;
	aux9 = Z2*Z2;

	R11 = pow((aux6 + aux8), 0.5);
	R12 = pow((aux6 + aux9), 0.5);
	R21 = pow((aux7 + aux8), 0.5);
	R22 = pow((aux7 + aux9), 0.5);

	aux10 = log((((R11 - d1)/(R11 + d1)) + EVITA_SINGULARIDADE_NUMERICA));
	aux11 = log((((R12 - d1)/(R12 + d1)) + EVITA_SINGULARIDADE_NUMERICA));
	aux12 = log((((R21 - d2)/(R21 + d2)) + EVITA_SINGULARIDADE_NUMERICA));
	aux13 = log((((R22 - d2)/(R22 + d2)) + EVITA_SINGULARIDADE_NUMERICA));
	aux14 = (double)(1.0/(2*d1));
	aux15 = (double)(1.0/(2*d2));
	aux16 = aux15*(aux13 - aux12);

	gyzaux += (X2*(1.0 + (m*m)) + c*m)*aux16;

	aux16 = aux14*(aux11 - aux10);

	gyzaux -= (X1*(1.0 + (m*m)) + c*m)*aux16;

	aux1 = (double)(1.0/(1.0 + (m*m)));

	gyzaux *= aux1;

	return gyzaux;

}

double integral_aresta_gzz (double X1, double X2, double Y1, double Y2, double Z1, double Z2) {

	double n, p, m, c, g, R11, R12, R21, R22, d1, d2;
	double aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
	double aux9, aux10, aux11, aux12, aux13, aux14, aux15, aux16;
	double gzzaux;

	gzzaux = 0.0;

	aux0 = X2 - X1 + EVITA_SINGULARIDADE_NUMERICA;
	aux1 = Y2 - Y1 + EVITA_SINGULARIDADE_NUMERICA;

	n = (double)(aux0/aux1);
	g = X1 - (Y1*n);

	m = (double)(aux1/aux0);
	c = Y1 - (X1*m);

	aux2 = pow(((aux0*aux0) + (aux1*aux1)), 0.5);
	aux3 = (X1*Y2) - (X2*Y1);

	p = ((double)(aux3/aux2)) + EVITA_SINGULARIDADE_NUMERICA;

	aux4 = (aux0*X1) + (aux1*Y1);
	aux5 = (aux0*X2) + (aux1*Y2);

	d1 = ((double)(aux4/aux2)) + EVITA_SINGULARIDADE_NUMERICA;
	d2 = ((double)(aux5/aux2)) + EVITA_SINGULARIDADE_NUMERICA;

	aux6 = (X1*X1) + (Y1*Y1);
	aux7 = (X2*X2) + (Y2*Y2);
	aux8 = Z1*Z1;
	aux9 = Z2*Z2;

	R11 = pow((aux6 + aux8), 0.5);
	R12 = pow((aux6 + aux9), 0.5);
	R21 = pow((aux7 + aux8), 0.5);
	R22 = pow((aux7 + aux9), 0.5);

	aux10 = atan((double)((Z2*d2)/(p*R22)));
	aux11 = atan((double)((Z1*d2)/(p*R21)));
	aux12 = aux10 - aux11;

	gzzaux += aux12;

	aux10 = atan((double)((Z2*d1)/(p*R12)));
	aux11 = atan((double)((Z1*d1)/(p*R11)));
	aux12 = aux10 - aux11;

	gzzaux -= aux12;

	return gzzaux;

}
