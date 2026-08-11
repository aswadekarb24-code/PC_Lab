#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[])
{
    // 1. Check for correct command line arguments
    // argc is 4 because: argv[0]=program, argv[1]=m, argv[2]=n, argv[3]=p
    if (argc != 4)
    {
        printf("Usage: %s <rows_m> <cols_n> <cols_p>\n", argv[0]);
        return -1;
    }

    int i, j, k;
    
    // Parse the command line arguments
    int m = atoi(argv[1]); // Rows in Matrix 1
    int n = atoi(argv[2]); // Cols in Matrix 1 / Rows in Matrix 2
    int p = atoi(argv[3]); // Cols in Matrix 2

    // Seed the random number generator so we get different numbers each run
    srand(time(NULL));

    // 2. Allocate and fill Matrix A dynamically
    int **matrixA = (int **)malloc(m * sizeof(int *));
    for (i = 0; i < m; i++)
    {
        matrixA[i] = (int *)malloc(n * sizeof(int));
        for (j = 0; j < n; j++)
        {
            matrixA[i][j] = rand() % 100; // Random integers from 0-99
        }
    }

    // 3. Allocate and fill Matrix B dynamically
    int **matrixB = (int **)malloc(n * sizeof(int *));
    for (i = 0; i < n; i++)
    {
        matrixB[i] = (int *)malloc(p * sizeof(int));
        for (j = 0; j < p; j++)
        {
            matrixB[i][j] = rand() % 100; // Random integers from 0-99
        }
    }

    // 4. Allocate Matrix C dynamically
    int **matrixC = (int **)malloc(m * sizeof(int *));
    for (i = 0; i < m; i++)
    {
        matrixC[i] = (int *)malloc(p * sizeof(int));
    }

    // ==========================================
    // MULTIPLICATION & BENCHMARKING
    // ==========================================
    double start_time = omp_get_wtime();

    {
        for (i = 0; i < m; i = i + 1)
        {
            for (j = 0; j < p; j = j + 1)
            {
                matrixC[i][j] = 0;
                for (k = 0; k < n; k = k + 1)
                {
                    matrixC[i][j] = (matrixC[i][j]) + ((matrixA[i][k]) * (matrixB[k][j]));
                }
            }
        }
    }

    double end_time = omp_get_wtime();
    // ==========================================

    // 5. Print results (only if matrix is small enough to reasonably view)
    if (m <= 10 && p <= 10) 
    {
        printf("The output after Matrix Multiplication is: \n");
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < p; j++)
                printf("%d \t", matrixC[i][j]);
            printf("\n");
        }
    } 
    else 
    {
        printf("Matrices successfully multiplied.\n");
        printf("(Output suppressed because matrix dimensions are > 10x10)\n");
    }

    printf("Time taken: %f seconds\n", end_time - start_time);

    // 6. Free the dynamically allocated memory
    for(i = 0; i < m; i++) {
        free(matrixA[i]);
        free(matrixC[i]);
    }
    free(matrixA);
    free(matrixC);

    for(i = 0; i < n; i++) {
        free(matrixB[i]);
    }
    free(matrixB);

    return 0;
}