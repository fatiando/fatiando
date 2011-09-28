#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EVITA_SINGULARIDADE_NUMERICA 1E-15

#define Divide_macro(a, b) ((a)/((b) + (1E-10)))
#define Divide_macro2(a, b) (((a) + (1E-10))/((b) + (1E-10)))

double **aloca_matriz_double_var (FILE *, int, int *);

double **aloca_matriz_double (FILE *, int, int);

double **libera_matriz_double (int, double **);

double *aloca_vetor_double (FILE *, int);

double *libera_vetor_double (double *);

int *aloca_vetor_int (FILE *, int);

int *libera_vetor_int (int *);

double cilin_pol_mag (int N, int M, int *nvertices, double *Xp, double *Yp, double *Zp, double *T_calc, double **J, double Bx_geomagnetico, double By_geomagnetico, double Bz_geomagnetico, double *z1, double *z2, double **x, double **y);

int paraview (int, int, int *, double **, double **, double *, double *);

double integral_aresta_V1 (double X1, double X2, double Y1, double Y2, double Z1, double Z2);

double integral_aresta_V2 (double X1, double X2, double Y1, double Y2, double Z1, double Z2);

double integral_aresta_V3 (double X1, double X2, double Y1, double Y2, double Z1, double Z2);

double integral_aresta_V4 (double X1, double X2, double Y1, double Y2, double Z1, double Z2);

double integral_aresta_V5 (double X1, double X2, double Y1, double Y2, double Z1, double Z2);

double integral_aresta_V6 (double X1, double X2, double Y1, double Y2, double Z1, double Z2);

int main() {

	int i, j, k;
	int N, M, Q;

	double **x, **y, *z1, *z2;
	double **J, intensidade_geomagnetica, inclinacao_geomagnetica, declinacao_geomagnetica, Bx_geomagnetico, By_geomagnetico, Bz_geomagnetico;
	int *nvertices;
    double *Xp, *Yp, *Zp, *T_calc;
    double aux0, aux1, aux2, aux3, aux4, aux5;

    /*

    N: número de pontos onde se deseja calcular a força magnética.
    M: número de prismas.
    **x: Matriz que armazena as coordenadas x dos vértices dos polígonos. A posição
        ij armazena a coordenada x do j-ésimo vértice do i-ésimo polígono.
    **y: Matriz que armazena as coordenadas y dos vértices dos polígonos. A posição
        ij armazena a coordenada y do j-ésimo vértice do i-ésimo polígono.
    **J: Matriz que armazena as componentes do vetor de magnetização de cada prisma.
        A primeira coluna é a intensidade de magnetização, a segunda é a declinação magnética
		e a terceira é a inclinação magnética.
    *nvertices: Vetor que armazena o número de vértices de cada prisma.
    *z1, *z2: Vetores que armazenam a coordenada z do topo e da base de cada prisma, respectivamente.
    *Xp, *Yp e *Zp: Vetores que armazenam as coordenadas x, y e z, respectivamente, dos
        pontos onde se deseja calcular a força magnética.
    *T_cal: Vetor que armazena o campo total observado.

    */

	char str[100];

	FILE *relatorio, *entrada, *saida;

	relatorio = fopen ("relatorio.txt", "w");

	/******** leitura do arquivo do prisma ==> **************/

	sprintf (str, "modelo-sintetico.txt");

	if (fopen(str, "r") == NULL) {

		fprintf (relatorio, "Arquivo %s nao encontrado!\n\n", str);

		fclose (relatorio);

		printf ("Erro!\n\n");

		system ("PAUSE");

		return 0;

	}

	entrada = fopen(str, "r");

	fscanf(entrada, "%d", &M);

	nvertices = aloca_vetor_int (relatorio, M);
	J = aloca_matriz_double (relatorio, M, 3);
	z1 = aloca_vetor_double (relatorio, M);
	z2 = aloca_vetor_double (relatorio, M);

	Q = 0;

	for (i = 0; i < M; i++) {

		if (fscanf(entrada, "%d", &nvertices[i]) != 1) {

			fprintf(relatorio, "Erro na leitura do arquivo %s!\n\n", str);

			fclose (relatorio);

			printf ("Erro!\n\n");

			system ("PAUSE");

			return 0;

		}

		Q += nvertices[i];

	}

	x = aloca_matriz_double_var (relatorio, M, nvertices);
	y = aloca_matriz_double_var (relatorio, M, nvertices);

	fscanf(entrada, "%lf %lf %lf", &intensidade_geomagnetica, &declinacao_geomagnetica, &inclinacao_geomagnetica);

	declinacao_geomagnetica = (double)(3.14159265358979323846*declinacao_geomagnetica/180.0);
	inclinacao_geomagnetica = (double)(3.14159265358979323846*inclinacao_geomagnetica/180.0);

	aux0 = intensidade_geomagnetica*cos(inclinacao_geomagnetica);
	
	Bx_geomagnetico = aux0*cos(declinacao_geomagnetica);
	By_geomagnetico = aux0*sin(declinacao_geomagnetica);
	Bz_geomagnetico = intensidade_geomagnetica*sin(inclinacao_geomagnetica);

	for (i = 0; i < M; i++) {

        if(fscanf(entrada, "%lf %lf %lf %lf %lf", &J[i][0], &J[i][1], &J[i][2], &z1[i], &z2[i]) != 5) {

			fprintf(relatorio, "Erro na leitura do arquivo %s!\n\n", str);

			fclose (relatorio);

			printf ("Erro!\n\n");

			system ("PAUSE");

			return 0;

        }

		/* Constante de proporcionalidade no SI */
		J[i][0] *= 1E-7;

		J[i][1] = (double)(3.14159265358979323846*J[i][1]/180.0);
		J[i][2] = (double)(3.14159265358979323846*J[i][2]/180.0);

		aux0 = J[i][0]*cos(J[i][2])*cos(J[i][1]);
		aux1 = J[i][0]*cos(J[i][2])*sin(J[i][1]);
		aux2 = J[i][0]*sin(J[i][2]);

		J[i][0] = aux0;
		J[i][1] = aux1;
		J[i][2] = aux2;

        for (j = 0; j < nvertices[i]; j++) {

			if (fscanf(entrada, "%lf %lf", &y[i][j], &x[i][j]) != 2) {

				fprintf(relatorio, "Erro na leitura do arquivo %s!\n\n", str);

				fclose (relatorio);

				printf ("Erro!\n\n");

				system ("PAUSE");

				return 0;

			}

        }

	}

	fclose (entrada);

	/******** <== leitura do arquivo do prisma **************/

	/******** leitura do arquivo de pontos ==> **************/

	sprintf (str, "pontos.txt");

	if (fopen(str, "r") == NULL) {

		fprintf (relatorio, "Arquivo %s nao encontrado!\n\n", str);

		fclose (relatorio);

		printf ("Erro!\n\n");

		system ("PAUSE");

		return 0;

	}

	entrada = fopen(str, "r");

	fscanf(entrada, "%d", &N);

	Xp = aloca_vetor_double(relatorio, N);
	Yp = aloca_vetor_double(relatorio, N);
	Zp = aloca_vetor_double(relatorio, N);
	T_calc = aloca_vetor_double(relatorio, N);

	for (i = 0; i < N; i++) {

		if (fscanf(entrada, "%lf %lf %lf", &Yp[i], &Xp[i], &Zp[i]) != 3) {

			fprintf(relatorio, "Erro na leitura do arquivo %s!\n\n", str);

			fclose (relatorio);

			printf ("Erro!\n\n");

			system ("PAUSE");

			return 0;

		}

	}

	fclose (entrada);

	/******** <== leitura do arquivo de pontos **************/

	sprintf (str, "anomalia.txt");

	cilin_pol_mag (N, M, nvertices, Xp, Yp, Zp, T_calc, J, Bx_geomagnetico, By_geomagnetico, Bz_geomagnetico, z1, z2, x, y);

	/********* impressão do arquivo de saida ==> ************/

	saida = fopen(str, "w");

	fprintf (saida, "              Y               X               Z     Campo-Total\n\n");

	for (i = 0; i < N; i++) {

		fprintf (saida, "%15.3lf %15.3lf %15.3lf %15.5lf\n", Yp[i], Xp[i], Zp[i], T_calc[i]);

	}

    fprintf (saida, "\n\n");

	fclose (saida);

    paraview (M, Q, nvertices, x, y, z1, z2);

	fclose (relatorio);

	printf ("Programa finalizado com sucesso!\n\n");

	system ("PAUSE");

	return 0;

}

double **aloca_matriz_double_var (FILE *arq, int linha, int *coluna) {
      
    double **m;  /* ponteiro para a matriz */
    int   i, j;
     
    /* aloca as linhas da matriz */
     
    m = (double **)calloc(linha, sizeof(double *));
     
    if (m == NULL) {
         
        fprintf (arq, "Memoria Insuficiente (linhas)!\n\n");
        
        fclose (arq);
        
        system ("PAUSE");
        return 0;
         
        return (NULL);
             
    }
     
    /* aloca as colunas da matriz */
     
    for ( i = 0; i < linha; i++ ) {

		j = coluna[i];

        m[i] = (double *)calloc(j, sizeof(double));
         
        if (m[i] == NULL) {
             
            fprintf (arq, "Memoria Insuficiente (colunas)!\n\n");
            
            fclose (arq);
            
            system ("PAUSE");
            return 0;
             
            return (NULL);
         
        }
         
    }
      
    return (m); /* retorna o ponteiro para a matriz */
      
}

double **aloca_matriz_double (FILE *arq, int linha, int coluna) {

    double **m;  /* ponteiro para a matriz */
    int   i, j;

    /* aloca as linhas da matriz */

    m = (double **)calloc(linha, sizeof(double *));

    if (m == NULL) {

        fprintf (arq, "Memoria Insuficiente (linhas)!\n\n");

        fclose (arq);

        system ("PAUSE");
        return 0;

        return (NULL);

    }

    /* aloca as colunas da matriz */

    for ( i = 0; i < linha; i++ ) {

        m[i] = (double *)calloc(coluna, sizeof(double));

        if (m[i] == NULL) {

            fprintf (arq, "Memoria Insuficiente (colunas)!\n\n");

            fclose (arq);

            system ("PAUSE");
            return 0;

            return (NULL);

        }

    }

    return (m); /* retorna o ponteiro para a matriz */

}


double **libera_matriz_double (int linha, double **m) {
      
    int  i;  /* variavel auxiliar */
    
    if (m == NULL) { 
          
        return (NULL);
        
    }
    
    for (i = 0; i < linha; i++) { 
        
        free (m[i]); /* libera as linhas da matriz */
        
    }
    
    free (m); /* libera a matriz */
        
    return (NULL); /* retorna um ponteiro nulo */
    
}

double *aloca_vetor_double (FILE *arq, int tamanho) {
       
    double *v; /* ponteiro para o vetor */
    
    v = (double *)calloc(tamanho, sizeof(double));
        
    if (v == NULL) { /*** verifica se há memória suficiente ***/
          
        fprintf (arq, "Memoria Insuficiente!\n\n");
        
        return (NULL);
        
        fclose (arq);
        
        system ("PAUSE");
        return 0;
        
    }
     
    return (v); /* retorna o ponteiro para o vetor */
       			           
}

double *libera_vetor_double (double *v) {
      
    if (v == NULL) {
          
        return (NULL);
        
    }
    
    free(v); /* libera o vetor */
    
    return (NULL); /* retorna o ponteiro */
    
}

int *aloca_vetor_int (FILE *arq, int tamanho) {
       
    int *v; /* ponteiro para o vetor */
    
    v = (int *)calloc(tamanho, sizeof(int));
        
    if (v == NULL) { /*** verifica se há memória suficiente ***/
          
        fprintf (arq, "Memoria Insuficiente!\n\n");
        
        return (NULL);
        
        fclose (arq);
        
        system ("PAUSE");
        return 0;
        
    }
     
    return (v); /* retorna o ponteiro para o vetor */
       			           
}

int *libera_vetor_int (int *v) {
      
    if (v == NULL) {
          
        return (NULL);
        
    }
    
    free(v); /* libera o vetor */
    
    return (NULL); /* retorna o ponteiro */
    
}

double cilin_pol_mag (int N, int M, int *nvertices, double *Xp, double *Yp, double *Zp,
							double *T_calc, double **J, double Bx_geomagnetico, double By_geomagnetico, double Bz_geomagnetico,
							double *z1, double *z2, double **x, double **y) { 

	int i, j, k;
	double X1, X2, Y1, Y2, Z1, Z2;
	double V1, V2, V3, V4, V5, V6;
	double Bx, By, Bz;
	double aux0;

	/* Cálculo da força magnética ==> */

	for (i = 0; i < N; i++) { /* looping dos pontos */

		Bx = 0.0;
		By = 0.0;
		Bz = 0.0;

		for (j = 0; j < M; j++) { /* looping dos prismas */

			Z1 = z1[j] - Zp[i];
			Z2 = z2[j] - Zp[i];

			V1 = 0.0;
			V2 = 0.0;
			V3 = 0.0;
			V4 = 0.0;
			V5 = 0.0;
			V6 = 0.0;

			for (k = 0; k < (nvertices[j] - 1); k++) { /* looping dos vértices */

				X1 = x[j][k] - Xp[i];
				Y1 = y[j][k] - Yp[i];
				X2 = x[j][k+1] - Xp[i];
				Y2 = y[j][k+1] - Yp[i];

				V1 += integral_aresta_V1 (X1, X2, Y1, Y2, Z1, Z2);
				V2 += integral_aresta_V2 (X1, X2, Y1, Y2, Z1, Z2);
				V3 += integral_aresta_V3 (X1, X2, Y1, Y2, Z1, Z2);
				V4 += integral_aresta_V4 (X1, X2, Y1, Y2, Z1, Z2);
				V5 += integral_aresta_V5 (X1, X2, Y1, Y2, Z1, Z2);
				V6 += integral_aresta_V6 (X1, X2, Y1, Y2, Z1, Z2);

			} /* loooping dos vértices */

			/* cálculo referente a última aresta ==> */

				X1 = x[j][k] - Xp[i];
				Y1 = y[j][k] - Yp[i];
				X2 = x[j][0] - Xp[i];
				Y2 = y[j][0] - Yp[i];


				V1 += integral_aresta_V1 (X1, X2, Y1, Y2, Z1, Z2);
				V2 += integral_aresta_V2 (X1, X2, Y1, Y2, Z1, Z2);
				V3 += integral_aresta_V3 (X1, X2, Y1, Y2, Z1, Z2);
				V4 += integral_aresta_V4 (X1, X2, Y1, Y2, Z1, Z2);
				V5 += integral_aresta_V5 (X1, X2, Y1, Y2, Z1, Z2);
				V6 += integral_aresta_V6 (X1, X2, Y1, Y2, Z1, Z2);

			/* <== cálculo referente a última aresta */

			Bx += (J[j][0]*V1) + (J[j][1]*V2) + (J[j][2]*V3);
			By += (J[j][0]*V2) + (J[j][1]*V4) + (J[j][2]*V5);
			Bz += (J[j][0]*V3) + (J[j][1]*V5) + (J[j][2]*V6);

		} /* looping dos prismas */
		
		Bx *= 1E9;
		By *= 1E9;
		Bz *= 1E9;

		aux0 = pow((Bx_geomagnetico + Bx), 2) + pow((By_geomagnetico + By), 2) + pow((Bz_geomagnetico + Bz), 2);
		
		T_calc[i] = pow(aux0, 0.5);

		aux0 = pow(Bx_geomagnetico, 2) + pow(By_geomagnetico, 2) + pow(Bz_geomagnetico, 2);
		
		aux0 = pow(aux0, 0.5);
		
		T_calc[i] -= aux0;
		
	} /* looping dos pontos */

	/* <== Cálculo da força magnética */

	return 0;

}

int paraview (int M, int Q, int *nvertices, double **x, double **y, double *z1, double *z2) {

	int i, j, k;

	FILE *arquivo;

	arquivo = fopen("vert_true.vtk", "w");

	fprintf (arquivo, "# vtk DataFile Version 2.0\n");
	fprintf (arquivo, "Modelo Inverso\n");
	fprintf (arquivo, "ASCII\n");
	fprintf (arquivo, "DATASET POLYDATA\n");
	fprintf (arquivo, "POINTS %d float\n\n", (2*Q));

	for (i = 0; i < M; i++) {

		for (j = 0; j < nvertices[i]; j++) {

			fprintf (arquivo, "%15.5lf %15.5lf %15.5lf\n", x[i][j], y[i][j], z1[i]);

		}

		for (j = 0; j < nvertices[i]; j++) {

			/*fprintf (arquivo, "%15.5lf %15.5lf %15.5lf\n", aux2, aux1, z2[i]);*/
			fprintf (arquivo, "%15.5lf %15.5lf %15.5lf\n", x[i][j], y[i][j], (z1[i] + (0.98*(z2[i] - z1[i]))));

		}


	}

    fprintf(arquivo, "\n");
    fprintf (arquivo, "POLYGONS %d %d\n\n", (Q + M*2), (Q*7 + M*2));

	for (i = 0, k = 0; i < M; i++) {

		/* face de cima do i-ésimo prisma */
		fprintf (arquivo, "%3d ", nvertices[i]);

		for (j = k; j < (k + nvertices[i]); j++) {

			fprintf (arquivo, "%3d ", j);

		}

		fprintf (arquivo, "\n");

		/* face de baixo do i-ésimo prisma */
		fprintf (arquivo, "%3d ", nvertices[i]);

		for ( ; j < (k + (2*nvertices[i])); j++) {

			fprintf (arquivo, "%3d ", j);

		}

		fprintf (arquivo, "\n");

        k += 2*nvertices[i];

	}

   	for (i = 0, k = 0; i < M; i++) {

		/* faces laterais do i-ésimo prisma */
		for (j = k; j <  (k + nvertices[i] - 1); j++) {

			fprintf (arquivo, "  4 ");
			fprintf (arquivo, "%3d ", j);
			fprintf (arquivo, "%3d ", (j + nvertices[i]));
			fprintf (arquivo, "%3d ", (j + nvertices[i] + 1));
			fprintf (arquivo, "%3d\n", (j + 1));

		}

			fprintf (arquivo, "  4 ");
			fprintf (arquivo, "%3d ", j);
			fprintf (arquivo, "%3d ", (j + nvertices[i]));
			fprintf (arquivo, "%3d ", (k + nvertices[i]));
			fprintf (arquivo, "%3d\n", k);

		k += 2*nvertices[i];

	}

	fclose(arquivo);

	return 0;

}

double integral_aresta_V1 (double X1, double X2, double Y1, double Y2, double Z1, double Z2) {

	double n, p, m, c, g, R11, R12, R21, R22, d1, d2;
	double aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
	double aux9, aux10, aux11, aux12, aux13, aux14, aux15, aux16;
	double V1aux;

	V1aux = 0.0;

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

	V1aux += (g*Y2*aux13) + (n*aux14);

	aux10 = atan((double)((Z2*d1)/(p*R12)));
	aux11 = atan((double)((Z1*d1)/(p*R11)));
	aux12 = aux10 - aux11;
	aux13 = (double)(aux12/(p*d1));
	aux14 = (double)((p*aux12)/d1);

	V1aux -= (g*Y1*aux13) + (n*aux14);

	aux10 = log(((Z2 + R22) + EVITA_SINGULARIDADE_NUMERICA));
	aux11 = log(((Z1 + R21) + EVITA_SINGULARIDADE_NUMERICA));
	aux12 = log(((Z2 + R12) + EVITA_SINGULARIDADE_NUMERICA));
	aux13 = log(((Z1 + R11) + EVITA_SINGULARIDADE_NUMERICA));
	aux14 = aux10 - aux11;
	aux15 = aux12 - aux13;

	V1aux += (n*(aux15 - aux14));

	aux0 = (double)(1.0/(1.0 + (n*n)));
	
	V1aux *= -aux0;

	return V1aux;

}

double integral_aresta_V2 (double X1, double X2, double Y1, double Y2, double Z1, double Z2) {

	double n, p, m, c, g, R11, R12, R21, R22, d1, d2;
	double aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
	double aux9, aux10, aux11, aux12, aux13, aux14, aux15, aux16;
	double V2aux;

	V2aux = 0.0;

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

	V2aux += (((g*g) + (g*n*Y2))*aux13) - aux14;

	aux10 = atan((double)((Z2*d1)/(p*R12)));
	aux11 = atan((double)((Z1*d1)/(p*R11)));
	aux12 = aux10 - aux11;
	aux13 = (double)(aux12/(p*d1));
	aux14 = (double)((p*aux12)/d1);

	V2aux -= (((g*g) + (g*n*Y1))*aux13) - aux14;

	aux10 = log(((Z2 + R22) + EVITA_SINGULARIDADE_NUMERICA));
	aux11 = log(((Z1 + R21) + EVITA_SINGULARIDADE_NUMERICA));
	aux12 = log(((Z2 + R12) + EVITA_SINGULARIDADE_NUMERICA));
	aux13 = log(((Z1 + R11) + EVITA_SINGULARIDADE_NUMERICA));
	aux14 = aux10 - aux11;
	aux15 = aux12 - aux13;

	V2aux += (aux14 - aux15);

	aux0 = (double)(1.0/(1.0 + (n*n)));
	
	V2aux *= aux0;

	return V2aux;

}

double integral_aresta_V3 (double X1, double X2, double Y1, double Y2, double Z1, double Z2) {

	double n, p, m, c, g, R11, R12, R21, R22, d1, d2;
	double aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
	double aux9, aux10, aux11, aux12, aux13, aux14, aux15, aux16;
	double V3aux;

	V3aux = 0.0;

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
		
	V3aux += (Y2*(1.0 + (n*n)) + g*n)*aux16;

	aux16 = aux14*(aux11 - aux10);

	V3aux -= (Y1*(1.0 + (n*n)) + g*n)*aux16;

	aux0 = (double)(1.0/(1.0 + (n*n)));
	
	V3aux *= -aux0;

	return V3aux;

}

double integral_aresta_V4 (double X1, double X2, double Y1, double Y2, double Z1, double Z2) {

	double n, p, m, c, g, R11, R12, R21, R22, d1, d2;
	double aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
	double aux9, aux10, aux11, aux12, aux13, aux14, aux15, aux16;
	double V4aux;

	V4aux = 0.0;

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

	V4aux += (c*X2*aux13) + (m*aux14);

	aux10 = atan((double)((Z2*d1)/(p*R12)));
	aux11 = atan((double)((Z1*d1)/(p*R11)));
	aux12 = aux10 - aux11;
	aux13 = (double)(aux12/(p*d1));
	aux14 = (double)((p*aux12)/d1);

	V4aux -= (c*X1*aux13) + (m*aux14);

	aux10 = log(((Z2 + R22) + EVITA_SINGULARIDADE_NUMERICA));
	aux11 = log(((Z1 + R21) + EVITA_SINGULARIDADE_NUMERICA));
	aux12 = log(((Z2 + R12) + EVITA_SINGULARIDADE_NUMERICA));
	aux13 = log(((Z1 + R11) + EVITA_SINGULARIDADE_NUMERICA));
	aux14 = aux10 - aux11;
	aux15 = aux12 - aux13;

	V4aux += (m*(aux15 - aux14));

	aux1 = (double)(1.0/(1.0 + (m*m)));
	
	V4aux *= aux1;

	return V4aux;

}

double integral_aresta_V5 (double X1, double X2, double Y1, double Y2, double Z1, double Z2) {

	double n, p, m, c, g, R11, R12, R21, R22, d1, d2;
	double aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
	double aux9, aux10, aux11, aux12, aux13, aux14, aux15, aux16;
	double V5aux;

	V5aux = 0.0;

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
				
	V5aux += (X2*(1.0 + (m*m)) + c*m)*aux16;

	aux16 = aux14*(aux11 - aux10);

	V5aux -= (X1*(1.0 + (m*m)) + c*m)*aux16;

	aux1 = (double)(1.0/(1.0 + (m*m)));
	
	V5aux *= aux1;

	return V5aux;

}

double integral_aresta_V6 (double X1, double X2, double Y1, double Y2, double Z1, double Z2) {

	double n, p, m, c, g, R11, R12, R21, R22, d1, d2;
	double aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
	double aux9, aux10, aux11, aux12, aux13, aux14, aux15, aux16;
	double V6aux;

	V6aux = 0.0;

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

	V6aux += aux12;

	aux10 = atan((double)((Z2*d1)/(p*R12)));
	aux11 = atan((double)((Z1*d1)/(p*R11)));
	aux12 = aux10 - aux11;

	V6aux -= aux12;

	return V6aux;

}