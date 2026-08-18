#include <stdio.h>
#include <omp.h>
int main()
{
    int num_steps = 10000000, i;
    double aux, pi, step = 1.0 / (double)num_steps, x = 0.0, sum = 0.0;
    double start_time = omp_get_wtime();
    omp_set_num_threads(8); // Set the number of threads to 16
#pragma omp parallel private(i, x, aux) shared(sum)
    {
#pragma omp for schedule(static)
        for (i = 0; i < num_steps; i = i + 1)
        {

            x = (i + 0.5) * step;
            aux = 4.0 / (1.0 + x * x);
#pragma omp critical
            sum = sum + aux;
        }
    }
    pi = step * sum;
    double end_time = omp_get_wtime();

    printf("The Value of PI is %.20lf\n", pi);
    printf("Time taken: %f seconds\n", end_time - start_time);

    return 0;
}