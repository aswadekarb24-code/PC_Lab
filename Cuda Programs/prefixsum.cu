/*********************************************************************************************
   THINK PARALLEL

  Name          : PrefixSum
  Objective     : Write a CUDA Program to perform PrefixSum
  Input         : None
  Output        : Execution time in seconds
  Created       : Hybrid Computing, 2013
********************************************************************************************/

#include<stdio.h>
#include<cuda.h>
#include <assert.h>
#include<sys/time.h>

#define N 5 

#define BLOCKSIZE 5

__global__ void PrefixSum(float *dInArray, float *dOutArray, int arrayLen, int threadDim)
{
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int tindex = (threadDim * tidx) + tidy;
	int maxNumThread = threadDim * threadDim; 
	int pass = 0;  
	int count ;
	int curEleInd;
	float tempResult = 0.0;

	while( (curEleInd = (tindex + maxNumThread * pass))  < arrayLen )
	{
		tempResult = 0.0f;
		for( count = 0; count <= curEleInd; count++)
			tempResult += dInArray[count];
		dOutArray[curEleInd] = tempResult;
		pass++;
	}
	__syncthreads();
}//end of Prefix sum function

void PrefixSum_cpu(float *x_h, float *z_h)
{
	int i;
	for(i=0; i<N; i++)
	{
		if(i==0)
			z_h[i]=x_h[i];
		else
			z_h[i]=z_h[i-1]+x_h[i];
	}
}

int main()
{
	float *x_h, *y_h, *z_h;
	float *x_d, *y_d;
	int i;

        struct timeval  TimeValue_Start;
        struct timezone TimeZone_Start;

        struct timeval  TimeValue_Final;
        struct timezone TimeZone_Final;
        long            time_start, time_end;
        double          time_overhead;


	size_t size = N*sizeof(float);

	x_h = (float *)malloc(size);
	y_h = (float *)malloc(size);
	z_h = (float *)malloc(size);

	cudaMalloc((void **)&x_d,size);
	cudaMalloc((void **)&y_d,size);

	for(i=0; i<N; i++)
	{
		x_h[i] = (float) i+1;
	}

	cudaMemcpy(x_d,x_h,size,cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
	dim3 dimGrid(1,1);

        gettimeofday(&TimeValue_Start, &TimeZone_Start);
	PrefixSum<<<dimGrid, dimBlock>>>(x_d, y_d, N, BLOCKSIZE);
        gettimeofday(&TimeValue_Final, &TimeZone_Final);


	cudaMemcpy(y_h,y_d,size,cudaMemcpyDeviceToHost);

	PrefixSum_cpu(x_h,z_h);

	for(i = 0; i < N; i++)
	{
	assert(y_h[i]==z_h[i]);
	printf("Prefix sum till index %d : %.3f\n", i, z_h[i]);
	}

        time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
        time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;

        time_overhead = (time_end - time_start)/1000000.0;

        printf("\nTime in Seconds (T): %lf\n\n",time_overhead);

	free(x_h);
	free(y_h);
	free(z_h);

	cudaFree(x_d);
	cudaFree(y_d);

return 0;
}
