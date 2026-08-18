#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

double func(double x) {
    return (4.0 / (1.0 + x * x));
}

int main(int argc, char *argv[]) {
    int NoInterval, interval;
    int MyRank, Numprocs, Root = 0;
    double mypi, pi, h, sum, x;
    double PI25DT = 3.141592653589793238462643;
    /* Turn off stdout buffering across MPI process handles */
    setvbuf(stdout, NULL, _IONBF, 0);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);

    

    if (MyRank == Root) {
        /* Included \n so mpirun flushes the line buffer to terminal immediately */
        printf("Enter the number of intervals:\n> ");
        fflush(stdout);
        if (scanf("%d", &NoInterval) != 1) {
            NoInterval = -1;
        }
    }

    /* Broadcast number of intervals to all processes */
    MPI_Bcast(&NoInterval, 1, MPI_INT, Root, MPI_COMM_WORLD);

    if (NoInterval <= 0) {
        if (MyRank == Root) {
            printf("Error: Invalid number of intervals.\n");
        }
        MPI_Finalize();
        return -1;
    }

    h = 1.0 / (double)NoInterval;
    sum = 0.0;

    for (interval = MyRank + 1; interval <= NoInterval; interval += Numprocs) {
        x = h * ((double)interval - 0.5);
        sum += func(x);
    }
    mypi = h * sum;

    MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, Root, MPI_COMM_WORLD);

    if (MyRank == Root) {
        printf("\n--- PI Calculation Results ---\n");
        printf("Total Processes : %d\n", Numprocs);
        printf("Interval Count  : %d\n", NoInterval);
        printf("Calculated PI   : %.16f\n", pi);
        printf("Exact Reference : %.16f\n", PI25DT);
        printf("Absolute Error  : %.16f\n", fabs(pi - PI25DT));
    }

    MPI_Finalize();
    return 0;
}