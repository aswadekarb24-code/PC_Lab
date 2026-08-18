#include <stdio.h>
#include "mpi.h"

int main(int argc, char *argv[]) {
    int MyRank, Numprocs, Root = 0;
    int value, sum = 0;
    int Source, Destination;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);

    /* Edge case: Single process execution */
    if (Numprocs == 1) {
        printf("Process 0 (Root): Running on 1 process. Final SUM = 0\n");
        MPI_Finalize();
        return 0;
    }

    if (MyRank == Root) {
        Destination = MyRank + 1;
        MPI_Send(&MyRank, 1, MPI_INT, Destination, 0, MPI_COMM_WORLD);

        Source = Numprocs - 1;
        MPI_Recv(&sum, 1, MPI_INT, Source, 0, MPI_COMM_WORLD, &status);
        printf("Process %d (Root): Ring completed across %d processes. Final SUM = %d\n", 
               MyRank, Numprocs, sum);
    } 
    else if (MyRank < Numprocs - 1) {
        Source = MyRank - 1;
        MPI_Recv(&value, 1, MPI_INT, Source, 0, MPI_COMM_WORLD, &status);

        sum = MyRank + value;
        Destination = MyRank + 1;
        MPI_Send(&sum, 1, MPI_INT, Destination, 0, MPI_COMM_WORLD);
    } 
    else {
        /* Last process (Numprocs - 1) routes token back to Root */
        Source = MyRank - 1;
        MPI_Recv(&value, 1, MPI_INT, Source, 0, MPI_COMM_WORLD, &status);

        sum = MyRank + value;
        Destination = Root;
        MPI_Send(&sum, 1, MPI_INT, Destination, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}