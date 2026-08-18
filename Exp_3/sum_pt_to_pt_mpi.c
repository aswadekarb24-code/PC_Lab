#include <stdio.h>
#include "mpi.h"

int main(int argc, char *argv[]) {
    int iproc;
    int MyRank, Numprocs, Root = 0;
    int value, sum = 0;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);

    if (MyRank == Root) {
        sum = MyRank; /* Include Root process value */
        for (iproc = 1; iproc < Numprocs; iproc++) {
            MPI_Recv(&value, 1, MPI_INT, iproc, 0, MPI_COMM_WORLD, &status);
            sum += value;
        }
        printf("Process %d (Root): Collected values from %d workers. Total SUM = %d\n", 
               MyRank, Numprocs - 1, sum);
    } 
    else {
        MPI_Send(&MyRank, 1, MPI_INT, Root, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}