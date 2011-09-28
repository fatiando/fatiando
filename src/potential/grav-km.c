#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define Divide_macro(a, b) ((a)/((b) + (1E-10)))
#define Divide_macro2(a, b) (((a) + (1E-10))/((b) + (1E-10)))

/* double GRAV = 0.0667; * (kilômetro cúbico)/((kilograma)*(segundo ao quadrado)) */
double GRAV = 6.67E-20; /* (kilômetro cúbico)/((kilograma)*(segundo ao quadrado)) */



double cilin_pol_vertical2 (int N, int M, int *nvertices, double *Xp, double *Yp, double *Zp,
							double *gz, double *rho, double *z1,
							double *z2, double **x, double **y) {

	int i, j, k;
	double gzaux, aux;
	double Xk1, Xk2, Yk1, Yk2, Z1, Z1_quadrado, Z2, Z2_quadrado;

	/******** Cálculo da atração gravitacional ==> **********/

	for (i = 0; i < N; i++) { /* <== for dos pontos */

		gz[i] = 0.0;

		for (j = 0; j < M; j++) { /* for dos prismas */

			aux = GRAV*rho[j]*1E12*1E8;
			/* o 1E12 transforma a densidade de g/cm^3 para kg/km^3 */
			/* o 1E8 transforma a aceleração gravitacional de km/s^2 para mGal */

			gzaux = 0.0;

			Z1 = z1[j] - Zp[i];
			Z2 = z2[j] - Zp[i];
			Z1_quadrado = pow(Z1, 2);
			Z2_quadrado = pow(Z2, 2);

			for (k = 0; k < (nvertices[j]-1); k++) { /* <== for dos vértices */

				Xk1 = x[j][k] - Xp[i];
				Xk2 = x[j][k+1] - Xp[i];
				Yk1 = y[j][k] - Yp[i];
				Yk2 = y[j][k+1] - Yp[i];

				gzaux += integral_aresta_gz (Xk1, Xk2, Yk1, Yk2, Z1, Z2, Z1_quadrado, Z2_quadrado);

			} /* <== for dos vértices */

			/***** última aresta ==> *******/

			Xk1 = x[j][nvertices[j]-1] - Xp[i];
			Xk2 = x[j][0] - Xp[i];
			Yk1 = y[j][nvertices[j]-1] - Yp[i];
			Yk2 = y[j][0] - Yp[i];

			gzaux += integral_aresta_gz (Xk1, Xk2, Yk1, Yk2, Z1, Z2, Z1_quadrado, Z2_quadrado);

			/***** <== última aresta *******/

			gzaux *= aux;

			gz[i] += gzaux;

		} /* for dos prismas */

	} /* <== for dos pontos */

	/******** <== Cálculo da atração gravitacional **********/

	return 0;

}

double integral_aresta_gz (double Xk1, double Xk2, double Yk1, double Yk2, double Z1, double Z2, double Z1_quadrado, double Z2_quadrado) {

	double gzaux, auxk1, auxk2, aux1k1, aux1k2, aux2k1, aux2k2;
	double Ak1, Ak2, Bk1, Bk2, Ck1, Ck2, Dk1, Dk2, E1k1, E1k2, E2k1, E2k2;
	double Qk1, Qk2, R1k1, R1k2, R2k1, R2k2, p, p_quadrado;

	gzaux = 0.0;

	p = (Xk1*Yk2) - (Xk2*Yk1);
	p_quadrado = pow(p, 2);
	Qk1 = ((Yk2 - Yk1)*Yk1) + ((Xk2 - Xk1)*Xk1);
	Qk2 = ((Yk2 - Yk1)*Yk2) + ((Xk2 - Xk1)*Xk2);

	Ak1 = pow(Xk1, 2) + pow(Yk1, 2);
	Ak2 = pow(Xk2, 2) + pow(Yk2, 2);
	R1k1 = Ak1 + Z1_quadrado;
	R1k1 = pow(R1k1, 0.5);
	R1k2 = Ak2 + Z1_quadrado;
	R1k2 = pow(R1k2, 0.5);
	R2k1 = Ak1 + Z2_quadrado;
	R2k1 = pow(R2k1, 0.5);
	R2k2 = Ak2 + Z2_quadrado;
	R2k2 = pow(R2k2, 0.5);

	Ak1 = pow(Ak1, 0.5);
	Ak2 = pow(Ak2, 0.5);

	Bk1 = pow(Qk1, 2) + p_quadrado;
	Bk1 = pow(Bk1, 0.5);
	Bk2 = pow(Qk2, 2) + p_quadrado;
	Bk2 = pow(Bk2, 0.5);

	Ck1 = Qk1*Ak1;
	Ck2 = Qk2*Ak2;

	Dk1 = Divide_macro(p, 2.0);
	Dk2 = Dk1;
	Dk1 *= Divide_macro(Ak1, Bk1);
	Dk2 *= Divide_macro(Ak2, Bk2);

	E1k1 = R1k1*Bk1;
	E1k2 = R1k2*Bk2;
	E2k1 = R2k1*Bk1;
	E2k2 = R2k2*Bk2;

	auxk1 = Divide_macro(Qk1, p);
	auxk2 = Divide_macro(Qk2, p);
	aux1k1 = Divide_macro(Z1, R1k1);
	aux1k2 = Divide_macro(Z1, R1k2);
	aux2k1 = Divide_macro(Z2, R2k1);
	aux2k2 = Divide_macro(Z2, R2k2);

	gzaux += (Z2 - Z1)*(atan(auxk2) - atan(auxk1));
	gzaux += Z2*(atan(aux2k1*auxk1) - atan(aux2k2*auxk2));
	gzaux += Z1*(atan(aux1k2*auxk2) - atan(aux1k1*auxk1));

	gzaux += Dk1*(log(Divide_macro2((E1k1 - Ck1), (E1k1 + Ck1))) - log(Divide_macro2((E2k1 - Ck1), (E2k1 + Ck1))));
	gzaux += Dk2*(log(Divide_macro2((E2k2 - Ck2), (E2k2 + Ck2))) - log(Divide_macro2((E1k2 - Ck2), (E1k2 + Ck2))));

	return gzaux;

}




int paraview (int M, int Q, int nvertices, double teta, double **raio, double *z1, double *z2, double *x0, double *y0, int contador_adaptativo) {

	int i, j, k, l;
	double aux0, aux1, aux2;
	double xmax, xmin, ymax, ymin, zmin, zmax, raiomax, x0min, x0max, y0min, y0max;
	char str[100];

	FILE *arquivo;

	raiomax = 0.0;
	x0min = x0[0];
	x0max = x0[0];
	y0min = y0[0];
	y0max = y0[0];

	sprintf(str, "vert_true_laterais%d.vtk", contador_adaptativo);

	arquivo = fopen(str, "w");

	fprintf (arquivo, "# vtk DataFile Version 2.0\n");
	fprintf (arquivo, "Modelo Inverso\n");
	fprintf (arquivo, "ASCII\n");
	fprintf (arquivo, "DATASET POLYDATA\n");
	fprintf (arquivo, "POINTS %d float\n\n", (2*Q));

	for (i = 0; i < M; i++) {

		for (j = 0; j < nvertices; j++) {

            aux0 = teta*j;

			aux1 = x0[i] - (raio[i][j]*sin(aux0));
			aux2 = y0[i] + (raio[i][j]*cos(aux0));

			fprintf (arquivo, "%15.5lf %15.5lf %15.5lf\n", aux1, aux2, z1[i]);

			/* cálculo dos limites da caixa */
			if (raio[i][j] > raiomax) {

				raiomax = raio[i][j];

			}
			if (x0min > x0[i]) {

				x0min = x0[i];

			}
			if (x0max < x0[i]) {

				x0max = x0[i];

			}
			if (y0min > y0[i]) {

				y0min = y0[i];

			}
			if (y0max < y0[i]) {

				y0max = y0[i];

			}

		}

		for (j = 0; j < nvertices; j++) {

            aux0 = teta*j;

			aux1 = x0[i] - (raio[i][j]*sin(aux0));
			aux2 = y0[i] + (raio[i][j]*cos(aux0));

			/*fprintf (arquivo, "%15.5lf %15.5lf %15.5lf\n", aux1, aux2, (z1[i] + (0.9*(z2[i] - z1[i]))));*/
			fprintf (arquivo, "%15.5lf %15.5lf %15.5lf\n", aux1, aux2, z2[i]);

		}


	}

    fprintf(arquivo, "\n");

	fprintf (arquivo, "POLYGONS %d %d\n\n", (Q), (Q*5));

   	for (i = 0, k = 0; i < M; i++) {

		/* faces laterais do i-ésimo prisma */
		for (j = k; j <  (k + nvertices - 1); j++) {

			fprintf (arquivo, "  4 ");
			fprintf (arquivo, "%3d ", j);
			fprintf (arquivo, "%3d ", (j + nvertices));
			fprintf (arquivo, "%3d ", (j + 1 + nvertices));
			fprintf (arquivo, "%3d\n", (j + 1));

		}

			fprintf (arquivo, "  4 ");
			fprintf (arquivo, "%3d ", j);
			fprintf (arquivo, "%3d ", (j + nvertices));
			fprintf (arquivo, "%3d ", (k + nvertices));
			fprintf (arquivo, "%3d\n", k);

		k += 2*nvertices;

	}

	fclose(arquivo);

	sprintf(str, "vert_true_tampas%d.vtk", contador_adaptativo);

	arquivo = fopen(str, "w");

	fprintf (arquivo, "# vtk DataFile Version 2.0\n");
	fprintf (arquivo, "Modelo Inverso\n");
	fprintf (arquivo, "ASCII\n");
	fprintf (arquivo, "DATASET POLYDATA\n");
	fprintf (arquivo, "POINTS %d float\n\n", (2*Q + 2*M));

	for (i = 0; i < M; i++) {

		/* face superior */
        fprintf (arquivo, "%15.5lf %15.5lf %15.5lf\n", x0[i], y0[i], z1[i]);

		for (j = 0; j < nvertices; j++) {

            aux0 = teta*j;

			aux1 = x0[i] - (raio[i][j]*sin(aux0));
			aux2 = y0[i] + (raio[i][j]*cos(aux0));

			fprintf (arquivo, "%15.5lf %15.5lf %15.5lf\n", aux1, aux2, z1[i]);

		}

		/* face inferior */
		/*fprintf (arquivo, "%15.5lf %15.5lf %15.5lf\n", x0[i], y0[i], (z1[i] + (0.98*(z2[i] - z1[i]))));*/
        fprintf (arquivo, "%15.5lf %15.5lf %15.5lf\n", x0[i], y0[i], z2[i]);

		for (j = 0; j < nvertices; j++) {

            aux0 = teta*j;

			aux1 = x0[i] - (raio[i][j]*sin(aux0));
			aux2 = y0[i] + (raio[i][j]*cos(aux0));

			/*fprintf (arquivo, "%15.5lf %15.5lf %15.5lf\n", aux1, aux2, (z1[i] + (0.98*(z2[i] - z1[i]))));*/
            fprintf (arquivo, "%15.5lf %15.5lf %15.5lf\n", aux1, aux2, z2[i]);

			/* cálculo dos limites da caixa */
			if (raio[i][j] > raiomax) {

				raiomax = raio[i][j];

			}
			if (x0min > x0[i]) {

				x0min = x0[i];

			}
			if (x0max < x0[i]) {

				x0max = x0[i];

			}
			if (y0min > y0[i]) {

				y0min = y0[i];

			}
			if (y0max < y0[i]) {

				y0max = y0[i];

			}

		}

	}

    fprintf(arquivo, "\n");

	fprintf (arquivo, "POLYGONS %d %d\n\n", (2*Q), (Q*2*4));

	for (i = 0, l = 0, k = (l + 1); i < M; i++) {

		/* face de cima do i-ésimo prisma */
		for (j = 0; j < (nvertices - 1); j++, k++) {

			fprintf (arquivo, "%6d ", 3);
			fprintf (arquivo, "%6d ", l);
			fprintf (arquivo, "%6d ", k);
			fprintf (arquivo, "%6d ", (k+1));
            fprintf (arquivo, "\n");

		}

		fprintf (arquivo, "%6d ", 3);
		fprintf (arquivo, "%6d ", l);
		fprintf (arquivo, "%6d ", k);
		fprintf (arquivo, "%6d ", (k - nvertices + 1));

		fprintf (arquivo, "\n");

		l += (nvertices + 1);
		k = (l + 1);

		/* face de baixo do i-ésimo prisma */
		for (j = 0; j < (nvertices - 1); j++, k++) {

			fprintf (arquivo, "%6d ", 3);
			fprintf (arquivo, "%6d ", l);
			fprintf (arquivo, "%6d ", k);
			fprintf (arquivo, "%6d  ", (k+1));
			fprintf (arquivo, "\n");

		}

		fprintf (arquivo, "%6d ", 3);
		fprintf (arquivo, "%6d ", l);
		fprintf (arquivo, "%6d ", k);
		fprintf (arquivo, "%6d ", (k - nvertices + 1));

		fprintf (arquivo, "\n");

		l += (nvertices + 1);
		k = (l + 1);

	}

	fclose (arquivo);

	sprintf(str, "caixa%d.vtk", contador_adaptativo);

	arquivo = fopen(str, "w");

    fprintf (arquivo, "# vtk DataFile Version 2.0\n");
	fprintf (arquivo, "Caixa\n");
	fprintf (arquivo, "ASCII\n");
	fprintf (arquivo, "DATASET POLYDATA\n");
	fprintf (arquivo, "POINTS 8 float\n\n");

	zmin = z1[0];
	zmax = 1.1*z2[M-1];
	xmin = x0min - (1.1*raiomax);
	xmax = x0max + (1.1*raiomax);
	ymin = y0min - (1.1*raiomax);
	ymax = y0max + (1.1*raiomax);

	fprintf (arquivo, "%10.3lf %10.3lf %10.3lf\n", xmin, ymin, zmin);
	fprintf (arquivo, "%10.3lf %10.3lf %10.3lf\n", xmax, ymin, zmin);
	fprintf (arquivo, "%10.3lf %10.3lf %10.3lf\n", xmax, ymax, zmin);
    fprintf (arquivo, "%10.3lf %10.3lf %10.3lf\n", xmin, ymax, zmin);
	fprintf (arquivo, "%10.3lf %10.3lf %10.3lf\n", xmin, ymin, zmax);
	fprintf (arquivo, "%10.3lf %10.3lf %10.3lf\n", xmax, ymin, zmax);
	fprintf (arquivo, "%10.3lf %10.3lf %10.3lf\n", xmax, ymax, zmax);
    fprintf (arquivo, "%10.3lf %10.3lf %10.3lf\n\n", xmin, ymax, zmax);

	fprintf (arquivo, "POLYGONS 2 10\n\n");
	fprintf (arquivo, "4 4 5 6 7\n");
	fprintf (arquivo, "4 2 3 7 6");

	fclose(arquivo);

	sprintf(str, "vert_true_contorno%d.vtk", contador_adaptativo);

	arquivo = fopen(str, "w");

	fprintf (arquivo, "# vtk DataFile Version 2.0\n");
	fprintf (arquivo, "Modelo Inverso\n");
	fprintf (arquivo, "ASCII\n");
	fprintf (arquivo, "DATASET POLYDATA\n");
	fprintf (arquivo, "POINTS %d float\n\n", (2*Q));

	for (i = 0; i < M; i++) {

		for (j = 0; j < nvertices; j++) {

            aux0 = teta*j;

			aux1 = x0[i] - (raio[i][j]*sin(aux0));
			aux2 = y0[i] + (raio[i][j]*cos(aux0));

			fprintf (arquivo, "%15.5lf %15.5lf %15.5lf\n", aux1, aux2, z1[i]);

		}

		for (j = 0; j < nvertices; j++) {

            aux0 = teta*j;

			aux1 = x0[i] - (raio[i][j]*sin(aux0));
			aux2 = y0[i] + (raio[i][j]*cos(aux0));

			/*fprintf (arquivo, "%15.5lf %15.5lf %15.5lf\n", aux1, aux2, (z1[i] + (0.9*(z2[i] - z1[i]))));*/
            fprintf (arquivo, "%15.5lf %15.5lf %15.5lf\n", aux1, aux2, z2[i]);

		}

	}

    fprintf(arquivo, "\n");

	fprintf (arquivo, "LINES %d %d\n\n", (Q*2), (Q*2*3));

	for (i = 0, k = 0; i < M; i++) {

		for (j = 0; j < (nvertices - 1); j++, k++) {

			fprintf (arquivo, "2 ");
			fprintf (arquivo, "%6d ", k);
			fprintf (arquivo, "%6d ", (k+1));
			fprintf (arquivo, "\n");

		}

		fprintf (arquivo, "2 ");
		fprintf (arquivo, "%6d ", k);
		fprintf (arquivo, "%6d ", (k - nvertices + 1));
		fprintf (arquivo, "\n");

		k++;

		for (j = 0; j < (nvertices - 1); j++, k++) {

			fprintf (arquivo, "2 ");
			fprintf (arquivo, "%6d ", k);
			fprintf (arquivo, "%6d ", (k+1));
            fprintf (arquivo, "\n");

		}

		fprintf (arquivo, "2 ");
		fprintf (arquivo, "%6d ", k);
		fprintf (arquivo, "%6d ", (k - nvertices + 1));
		fprintf (arquivo, "\n");

		k++;

	}

	fclose(arquivo);

	sprintf(str, "esqueleto_true%d.vtk", contador_adaptativo);

	arquivo = fopen(str, "w");

	fprintf (arquivo, "# vtk DataFile Version 2.0\n");
	fprintf (arquivo, "Modelo Inverso\n");
	fprintf (arquivo, "ASCII\n");
	fprintf (arquivo, "DATASET POLYDATA\n");
	fprintf (arquivo, "POINTS %d float\n\n", (2*Q));

	for (i = 0; i < M; i++) {

		for (j = 0; j < nvertices; j++) {

            aux0 = teta*j;

			aux1 = x0[i] - (raio[i][j]*sin(aux0));
			aux2 = y0[i] + (raio[i][j]*cos(aux0));

			fprintf (arquivo, "%15.5lf %15.5lf %15.5lf\n", aux1, aux2, z1[i]);

		}

		for (j = 0; j < nvertices; j++) {

            aux0 = teta*j;

			aux1 = x0[i] - (raio[i][j]*sin(aux0));
			aux2 = y0[i] + (raio[i][j]*cos(aux0));

			/*fprintf (arquivo, "%15.5lf %15.5lf %15.5lf\n", aux1, aux2, (z1[i] + (0.9*(z2[i] - z1[i]))));*/
            fprintf (arquivo, "%15.5lf %15.5lf %15.5lf\n", aux1, aux2, z2[i]);

		}

	}

    fprintf(arquivo, "\n");

	fprintf (arquivo, "LINES %d %d\n\n", Q, (Q*3));

	for (i = 0, k = 0; i < M; i++) {

		for (j = 0; j < (nvertices - 1); j++, k++) {

			fprintf (arquivo, "2 ");
			fprintf (arquivo, "%6d ", k);
			fprintf (arquivo, "%6d ", (k+1));
			fprintf (arquivo, "\n");

		}

		fprintf (arquivo, "2 ");
		fprintf (arquivo, "%6d ", k);
		fprintf (arquivo, "%6d ", (k - nvertices + 1));
		fprintf (arquivo, "\n");

		k++;

		for (j = 0; j < (nvertices - 1); j++, k++) {

			/*fprintf (arquivo, "2 ");
			fprintf (arquivo, "%6d ", k);
			fprintf (arquivo, "%6d ", (k+1));
            fprintf (arquivo, "\n");*/

		}

		/*fprintf (arquivo, "2 ");
		fprintf (arquivo, "%6d ", k);
		fprintf (arquivo, "%6d ", (k - nvertices + 1));
		fprintf (arquivo, "\n");*/

		k++;

	}

	fclose(arquivo);

	return 0;

}
