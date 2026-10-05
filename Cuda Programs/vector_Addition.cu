/*********************************************************************************************
   THINK PARALLEL

  Name          : Vector Addition
  Objective     : Write a CUDA Program to perform Vector Addition
  Input         : Threads per Block and Blocks per Grid
  Output        : Execution time in seconds
  Created       : Hybrid Computing, 2013
********************************************************************************************/

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>

#define N 4096 // size of array

__global__ void vectorAdd(int *a,int *b, int *c)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < N){
		c[tid] = a[tid]+b[tid];
	}
}

int main(int argc, char *argv[])
{
	int T = 10, B = 1; // threads per block and blocks per grid
	int a[N],b[N],c[N]; // vectors, statically declared
	int *dev_a, *dev_b, *dev_c;
	printf("Size of array = %d\n", N);
	do {
		printf("Enter number of threads per block (1024 max, comp. cap. 2.x ");
		scanf("%d",&T);
		printf("\nEnter number of blocks per grid: ");
		scanf("%d",&B);
		if (T * B < N) printf("Error T x B < N, try again\n");
	} while (T * B < N);

	cudaEvent_t start, stop; // using cuda events to measure time
	float elapsed_time_ms;

	cudaMalloc((void**)&dev_a,N * sizeof(int));
	cudaMalloc((void**)&dev_b,N * sizeof(int));
	cudaMalloc((void**)&dev_c,N * sizeof(int));

	for(int i=0;i<N;i++) { // load arrays with some numbers
		a[i] = i;
		b[i] = i*2;
	}	

	cudaMemcpy(dev_a, a , N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b , N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c , N*sizeof(int),cudaMemcpyHostToDevice);

	cudaEventCreate( &start ); // instrument code to measure start time
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );

	vectorAdd<<<B,T>>>(dev_a,dev_b,dev_c);

	cudaMemcpy(c,dev_c,N*sizeof(int),cudaMemcpyDeviceToHost);
	cudaEventRecord( stop, 0 ); // instrument code to measure end time

	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );

	for(int i=0;i<N;i++) {
		printf("%d+%d=%d\n",a[i],b[i],c[i]);
		assert(c[i]==(a[i]+b[i]));
	}
	
	printf("Time to calculate results: %f ms.\n", elapsed_time_ms);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	
	return 0;
}
