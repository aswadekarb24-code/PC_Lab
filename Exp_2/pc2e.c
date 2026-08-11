#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
int main()
{
    srand(time(NULL));
    int numberOfElements, currentMax = -1, iIterator, arrayInput[100000];
    printf("Enter the Number of Elements: ");
    scanf("%d", &numberOfElements);
    for (iIterator = 0; iIterator < numberOfElements; iIterator++)
    {
        if(numberOfElements < 10)scanf("%d", &arrayInput[iIterator]);
        else arrayInput[iIterator] = rand() % numberOfElements;
    }
    double start_time = omp_get_wtime();

#pragma omp parallel for shared(currentMax)
    for (iIterator = 0; iIterator < numberOfElements; iIterator++)
    {
#pragma omp critical
        if (arrayInput[iIterator] > currentMax)
        {
            currentMax = arrayInput[iIterator];
        }
    }
    double end_time = omp_get_wtime();

    printf("The Maximum Element is: %d\n", currentMax);
    printf("Time taken: %f seconds\n", end_time - start_time);
    return 0;
}