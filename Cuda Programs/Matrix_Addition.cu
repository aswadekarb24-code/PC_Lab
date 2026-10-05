/*********************************************************************************************
   THINK PARALLEL

  Name		: Matrix Addition
  Objective	: Write a CUDA Program to perform Matrix Matrix multiplication.
  Input		: None 
  Output	: Execution time in seconds 
  Created	: Hybrid Computing, 2013
*********************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <cuda.h>
#include <assert.h>

const int N = 4;
const int blocksize = 2;

__global__ void add_matrix_on_gpu( float* a, float *b, float *c, int N )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int index = i + j*N;
	if ( i < N && j < N )
		c[index] = a[index] + b[index];
}

void add_matrix_on_cpu(float *a, float *b, float *d)
{
	int i;
	for(i = 0; i < N*N; i++)
	d[i] = a[i]+b[i];
}

int main() 
{
	float *a = new float[N*N];
	float *b = new float[N*N];
	float *c = new float[N*N];
	float *d = new float[N*N];

	for ( int i = 0; i < N*N; ++i ) {
		a[i] = 1.0f; b[i] = 3.5f; }

	printf("Matrix A:\n");
	for(int i=0; i<N*N; i++)
	{
		printf("\t%f",a[i]);
		if((i+1)%N==0)
			printf("\n");
	}

	printf("Matrix B:\n");
	for(int i=0; i<N*N; i++)
	{
		printf("\t%f",b[i]);
		if((i+1)%N==0)
			printf("\n");
	}

        struct timeval  TimeValue_Start;
        struct timezone TimeZone_Start;

        struct timeval  TimeValue_Final;
        struct timezone TimeZone_Final;
        long            time_start, time_end;
        double          time_overhead;


	float *ad, *bd, *cd;
	const int size = N*N*sizeof(float);

	cudaMalloc( (void**)&ad, size );
	cudaMalloc( (void**)&bd, size );
	cudaMalloc( (void**)&cd, size );


	cudaMemcpy( ad, a, size, cudaMemcpyHostToDevice );
	cudaMemcpy( bd, b, size, cudaMemcpyHostToDevice );


	dim3 dimBlock( blocksize, blocksize );
	dim3 dimGrid( N/dimBlock.x, N/dimBlock.y );

	gettimeofday(&TimeValue_Start, &TimeZone_Start);
	add_matrix_on_gpu<<<dimGrid, dimBlock>>>( ad, bd, cd, N );
        gettimeofday(&TimeValue_Final, &TimeZone_Final);


	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost );
	
	add_matrix_on_cpu(a,b,d);

        time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
        time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;

        time_overhead = (time_end - time_start)/1000000.0;

	printf("result is:\n");
	for(int i=0; i<N*N; i++)
	{
		assert(c[i]==d[i]);
		printf("\t%f",d[i]);
		if((i+1)%N==0)
			printf("\n");
	}

	for(int i=0; i<N*N; i++)
	assert(c[i]==d[i]);

        printf("\nTime in Seconds (T): %lf\n\n",time_overhead);

	cudaFree( ad ); cudaFree( bd ); cudaFree( cd );
	delete[] a; delete[] b; delete[] c, delete[] d;
	return EXIT_SUCCESS;
}
