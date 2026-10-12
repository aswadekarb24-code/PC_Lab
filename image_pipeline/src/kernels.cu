#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// CUDA Kernel for parallel pixel blending (Alpha Blending / Cross Dissolve)
// imgA: Source image (BGR, 3 channels)
// imgB: Target image (BGR, 3 channels)
// output: Interpolated transition frame
// alpha: blending factor between 0.0 (imgA) and 1.0 (imgB)
__global__ void blendKernel(const unsigned char* __restrict__ imgA,
                            const unsigned char* __restrict__ imgB,
                            unsigned char* __restrict__ output,
                            float alpha, int totalPixels) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < totalPixels) {
        int pixelIdx = idx * 3; // B, G, R channels

        float invAlpha = 1.0f - alpha;

        output[pixelIdx + 0] = (unsigned char)(imgA[pixelIdx + 0] * invAlpha + imgB[pixelIdx + 0] * alpha); // Blue
        output[pixelIdx + 1] = (unsigned char)(imgA[pixelIdx + 1] * invAlpha + imgB[pixelIdx + 1] * alpha); // Green
        output[pixelIdx + 2] = (unsigned char)(imgA[pixelIdx + 2] * invAlpha + imgB[pixelIdx + 2] * alpha); // Red
    }
}

// Host wrapper function to launch the CUDA kernel safely
extern "C" void launchBlendKernel(const unsigned char* d_imgA, const unsigned char* d_imgB,
                                    unsigned char* d_output, float alpha, int width, int height) {
    int totalPixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    blendKernel<<<blocksPerGrid, threadsPerBlock>>>(d_imgA, d_imgB, d_output, alpha, totalPixels);
    cudaDeviceSynchronize();
}