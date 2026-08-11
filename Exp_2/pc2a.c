#include <stdio.h>
#include <omp.h>
int main()
{
    double start_time = omp_get_wtime();
    int nthreads, tid;
    /* Fork a team of threads giving them their own copies of variables */
    omp_set_num_threads(10);
#pragma omp parallel private(nthreads, tid)
    {
        /* Obtain thread number */
        tid = omp_get_thread_num();
        printf("Hello World from thread = %d\n", tid);
        /* Only master thread does this */
        if (tid == 0)
        {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }
    } /* All threads join master thread and disband */
    double end_time = omp_get_wtime();
    printf("Time taken: %f seconds\n", end_time - start_time);
}