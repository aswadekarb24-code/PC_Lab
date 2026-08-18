#include <stdio.h>
#include <omp.h>
int IsPrime(int number)
{
    int i;
    for (i = 2; i < number; i++)
    {
        if (number % i == 0 && i != number)
            return 0;
    }
    return 1;
}
int main()
{
    int noOfThreads, valueN, indexCount = 0, arrayVal[100000], tempValue;
    printf("Enter the Number of threads: ");
    scanf("%d", &noOfThreads);
    printf("Enter the value of N: ");
    scanf("%d", &valueN);
    omp_set_num_threads(noOfThreads);
    double start_time = omp_get_wtime();

// #pragma omp parallel for reduction(+ : indexCount)
    for (tempValue = 2; tempValue <= valueN; tempValue++)
    {
        if (IsPrime(tempValue))
        {
            arrayVal[indexCount] = tempValue;
            indexCount++;
        }
    }
    double end_time = omp_get_wtime();

    printf("Number of prime numbers between 2 and %d: %d\n", valueN, indexCount);
    printf("Time taken: %f seconds\n", end_time - start_time);
}