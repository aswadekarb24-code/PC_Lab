/******************************************************************************
	THINK PARALLEL

	Name	: Vector Multiplication
	Objective	: Write a CUDA program to perform Vector Multiplication
	Input	: None
	Output	: Execution time in seconds
	Created	: Hybrid Computing, 2013
*******************************************************************************/

#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>
#include <assert.h>

__global__ void mult_vect(float *x, float *y, float *z, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
	{
		z[idx] = x[idx] * y[idx];
	}
}

int main()
{
	float *x_h, *y_h, *z_h;
	float *x_d, *y_d, *z_d;
	int n = 20, i;

	size_t size = n * sizeof(float);

	struct timeval TimeValue_Start;
	struct timezone TimeZone_Start;

	struct timeval TimeValue_Final;
	struct timezone TimeZone_Final;
	long time_start, time_end;
	double time_overhead;

	/* allocating memory on CPU */
	x_h = (float *)malloc(size);
	y_h = (float *)malloc(size);
	z_h = (float *)malloc(size);

	/* allocating memory on Device */
	cudaMalloc((void **)&x_d, size);
	cudaMalloc((void **)&y_d, size);
	cudaMalloc((void **)&z_d, size);

	/* Initialization of Vectors */
	for (i = 0; i < n; i++)
	{
		x_h[i] = (float)i;
		y_h[i] = (float)i;
	}

	/* Copying from Host to Device */
	cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(z_d, z_h, size, cudaMemcpyHostToDevice);

	int block_size = 4;
	int num_blocks = (n + block_size - 1) / block_size;

	gettimeofday(&TimeValue_Start, &TimeZone_Start);

	/* kernel launching */
	mult_vect<<<num_blocks, block_size>>>(x_d, y_d, z_d, n);

	gettimeofday(&TimeValue_Final, &TimeZone_Final);

	/* Copying from Device to Host */
	cudaMemcpy(x_h, x_d, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_h, y_d, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(z_h, z_d, size, cudaMemcpyDeviceToHost);

	time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
	time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;

	time_overhead = (time_end - time_start) / 1000000.0;

	/* Checking whether the result is correct or not */
	for (i = 0; i < n; i++)
	{
		printf("%f ", z_h[i]);
		assert(z_h[i] == (x_h[i] * y_h[i]));
		if (z_h[i] == (x_h[i] * y_h[i]))
			printf("Success\n");
		else
			printf("Fail\n");
	}

	printf("\nTime in Seconds (T): %lf\n\n", time_overhead);

	free(x_h);
	free(y_h);
	free(z_h);

	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(z_d);
	return 0;
}
