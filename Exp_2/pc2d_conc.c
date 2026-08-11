#include <stdio.h>
#include <string.h>
#include <omp.h>
int main()
{
    int valueN, indexCount = 0, arrayVal[1000000], is_p[1000001], tempValue;
    printf("Enter the value of N: ");
    scanf("%d", &valueN);
    memset(is_p, 1,sizeof(is_p));
    double start_time = omp_get_wtime();
    is_p[0]=is_p[1]=0;
    for(tempValue = 2; tempValue <= valueN; tempValue++){
        if(is_p[tempValue]){
            arrayVal[indexCount++] = tempValue;
            for(int j = 2*tempValue; j<=valueN; j+= tempValue)is_p[j]=0;
        }
    }
    double end_time = omp_get_wtime();

    printf("Number of prime numbers between 2 and %d: %d\n", valueN, indexCount);
    printf("Time taken: %f seconds\n", end_time - start_time);
}