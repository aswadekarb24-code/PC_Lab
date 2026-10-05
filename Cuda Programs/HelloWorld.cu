#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>
#include <assert.h>
__global__ void kernel (void){}
int main(void){
kernel<<<1, 1>>>();
printf("Hello, World");
return 0;
}