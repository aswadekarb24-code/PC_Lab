/*****************************************************************************************
   THINK PARALLEL
Name		: Square of Array
Objective	: Findout the bug in the program and fix it
Input		: None
Output		: Print elements of array along with square of elements
Created		: Hybrid Computing,2013
******************************************************************************************/
#include <stdio.h>
#include <cuda.h>
 
// Kernel that executes on the CUDA device
__global__ void square_array(float *a, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx<N)
		a[idx] = a[idx] * a[idx];
}


// main routine that executes on the host
int main(void)
{
	float *a_h, *a_d, *b_h;  // Pointer to host & device arrays
	const int N = 10;  // Number of elements in arrays
	size_t size = N * sizeof(float);

	a_h = (float *)malloc(size);
	b_h = (float *)malloc(size);

	a_d = (float *)malloc(size);

	// Initialize host array and copy it to CUDA device
	for (int i=0; i<N; i++)
		a_h[i] = (float)i;

	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

	// Do calculation on device:
	int block_size = 4;
	int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);

	square_array <<< n_blocks, block_size >>> (a_d, N);

  	// Retrieve result from device and store it in host array
	cudaMemcpy(b_h, a_d, sizeof(float)*N, cudaMemcpyDeviceToHost);

	// Print results
	for (int i=0; i<N; i++)
	printf("%f %f\n", a_h[i], b_h[i]);

	// Cleanup
	free(a_h);
	cudaFree(a_d);
return 0;
}
