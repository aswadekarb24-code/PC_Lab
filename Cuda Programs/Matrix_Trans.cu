/*********************************************************************************************
   THINK PARALLEL

  Name		: Matrix Transpose
  Objective	: Write a CUDA Program to perform Matrix Transpose.
  Input		: None 
  Output	: Execution time in seconds 
  Created	: Hybrid Computing, 2013
********************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <assert.h>
#include <sys/time.h>

const int N = 8;
const int blocksize = 4;

__global__ void transpose_naive( float *out, float *in, const int N ) {
	unsigned int xIdx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIdx = blockDim.y * blockIdx.y + threadIdx.y;
	if ( xIdx < N && yIdx < N ) {
		unsigned int idx_in = xIdx + N * yIdx;
		unsigned int idx_out = yIdx + N * xIdx;
		out[idx_out] = in[idx_in];
	}
}

void mat_trans_cpu(float *a, float *c)
{
	int mn    = N*N;      /* N rows and N columns */
	int q     = mn - 1;
	int i = 0;      /* Index of 1D array that represents the matrix */

	do
	{
		int k = (i*N) % q;
		while (k>i)
		k = (N*k) % q;

		if (k!=i)
		{
			c[k] = a[i];
			c[i] = a[k];
		}
		else
			c[i] = a[i];
	} while ( ++i <= (mn -2) );
c[i]=a[i];
	/* Update row and column */
/*	matrix.M = N;
	matrix.N = M;*/
}

int main() 
{
	float *a = new float[N*N];
	float *b = new float[N*N];
	float *c = new float[N*N];

	int i;
	for ( i = 0; i < N*N; ++i ) {
		a[i] = drand48(); 
	}
        struct timeval  TimeValue_Start;
        struct timezone TimeZone_Start;

        struct timeval  TimeValue_Final;
        struct timezone TimeZone_Final;
        long            time_start, time_end;
        double          time_overhead;



	for ( i = 0; i < N*N; i++)
	{
		printf("\t%f",a[i]);
		if(((i+1)%N == 0))
		printf("\n");
	}

	float *ad, *bd ;
	const int size = N*N*sizeof(float);
	cudaMalloc( (void**)&ad, size );
	cudaMalloc( (void**)&bd, size );
	cudaMemcpy( ad, a, size, cudaMemcpyHostToDevice );

	dim3 dimBlock( blocksize, blocksize );
	dim3 dimGrid( N/dimBlock.x, N/dimBlock.y );

        gettimeofday(&TimeValue_Start, &TimeZone_Start);
	transpose_naive<<<dimGrid, dimBlock>>>( bd, ad, N );
        gettimeofday(&TimeValue_Final, &TimeZone_Final);

	cudaMemcpy( b, bd, size, cudaMemcpyDeviceToHost );
	mat_trans_cpu(a,c);	
/* To display uncomment this section */
	printf("result matrix\n");
	for ( i = 0; i < N*N; ++i ){
		printf("\t%f",c[i]);
		if( ((i+1)%N == 0))
		printf("\n");
	}


        time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
        time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;

        time_overhead = (time_end - time_start)/1000000.0;

	for(i=0; i<N*N; i++)
	assert(b[i]==c[i]);

        printf("\nTime in Seconds (T): %lf\n\n",time_overhead);


	cudaFree( ad ); cudaFree( bd );
	delete[] a; delete[] b, delete[] c; 
	return EXIT_SUCCESS;
}
