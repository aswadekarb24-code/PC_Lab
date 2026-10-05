#include <stdio.h>
#include <cuda.h>

__global__ void mult_vect(float  * x, float * y, float * z, int n)
{
	int idx= blockIdx.x * blockDim.x + threadIdx.x;
	if(idx <  n)
	{
		z[idx] = x[idx] * y[idx];
	}
}

int main()
{
	float *x_h, *y_h, *z_h;
	float *x_d, *y_d, *z_d;
	int n= 20,i;

	size_t size= n * sizeof(float);

	x_h= (float *)malloc(size);
	y_h= (float *)malloc(size);
	z_h= (float *)malloc(size);


	cudaMalloc( (void**)&x_d, size );
	cudaMalloc( (void**)&y_d, size );
	cudaMalloc( (void**)&z_d, size );

	for(i=0; i < n; i++)
	{
		x_h[i]= (float) i;
		y_h[i]= (float) i;
	}

	cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(z_d, z_h, size, cudaMemcpyHostToDevice);

	int block_size= 4;
	int num_blocks= (n + block_size - 1) / block_size;

	mult_vect <<<num_blocks,  block_size>>> (x_d, y_d, z_d, n);

	cudaMemcpy(x_h, x_d, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_h, y_d, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(z_h, z_d, size, cudaMemcpyDeviceToHost);

	for(i = 0; i < n ; i++)
	{
		printf("%f   ", z_h[i]);
		if(z_h[i] == (x_h[i] * y_h[i]))
			printf("Success\n");
		else
			printf("Fail\n");
	}

	free(x_h);
	free(y_h);
	free(z_h);

	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(z_d);
return 0;
}
