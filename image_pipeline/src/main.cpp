#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "common.hpp"

// Declare CUDA host wrapper function defined in kernels.cu
extern "C" void launchBlendKernel(const unsigned char* d_imgA, const unsigned char* d_imgB,
                                    unsigned char* d_output, float alpha, int width, int height);

int main(int argc, char** argv) {
    Options opt;
    bool showHelp = false;
    if (!parseArgs(argc, argv, opt, showHelp)) return 2;
    if (showHelp) { printUsage(argv[0]); return 0; }

    std::vector<std::string> imagePaths = listImages(opt.datasetPath);
    if (imagePaths.size() < 2) {
        std::cerr << "[ERROR] Need at least 2 images." << std::endl;
        return -1;
    }

    const int targetWidth = opt.width;
    const int targetHeight = opt.height;
    const int intermediateFrames = opt.intermediateFrames;
    const std::string& outputVideoPath = opt.outputPath;
    size_t imgSize = (size_t)targetWidth * targetHeight * 3;

    std::cout << "[INFO] Mode: parallel (OpenMP + CUDA)\n";
    std::cout << "[INFO] Found " << imagePaths.size() << " images. Resolution "
              << targetWidth << "x" << targetHeight << ", " << intermediateFrames
              << " intermediate frames.\n";

    // --- STEP 1: OpenMP Parallel Feature Extraction ---
    size_t numImages = imagePaths.size();
    std::vector<cv::Mat> histograms(numImages);
    std::vector<cv::Mat> preloadedImages(numImages);
    auto t1 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < numImages; ++i) {
        cv::Mat img = cv::imread(imagePaths[i]);
        if (!img.empty()) {
            cv::resize(img, img, cv::Size(targetWidth, targetHeight));
            preloadedImages[i] = img;
            histograms[i] = extractHSVHistogram(img);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    const double featureMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "[SUCCESS] [CO1 - OpenMP] Feature extraction completed in "
              << featureMs << " ms." << std::endl;

    // Drop images that failed to decode
    {
        size_t k = 0;
        for (size_t i = 0; i < numImages; ++i) {
            if (!preloadedImages[i].empty()) {
                if (k != i) {
                    preloadedImages[k] = preloadedImages[i];
                    histograms[k] = histograms[i];
                }
                ++k;
            }
        }
        if (k != numImages) {
            std::cerr << "[WARN] " << (numImages - k) << " image(s) could not be decoded and were skipped.\n";
            numImages = k;
            preloadedImages.resize(k);
            histograms.resize(k);
        }
        if (numImages < 2) {
            std::cerr << "[ERROR] Need at least 2 decodable images." << std::endl;
            return -1;
        }
    }

    // --- STEP 2: OpenMP NxN Similarity Matrix & TSP Sorting ---
    auto t3 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> similarityMatrix(numImages, std::vector<float>(numImages, 0.0f));

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < numImages; ++i) {
        for (size_t j = i + 1; j < numImages; ++j) {
            float score = (float)cv::compareHist(histograms[i], histograms[j], cv::HISTCMP_CORREL);
            similarityMatrix[i][j] = score;
            similarityMatrix[j][i] = score;
        }
    }

    // Greedy Nearest Neighbor Sort
    std::vector<int> orderedIndices = greedyOrder(similarityMatrix);
    auto t4 = std::chrono::high_resolution_clock::now();
    const double orderMs = std::chrono::duration<double, std::milli>(t4 - t3).count();
    std::cout << "[SUCCESS] [CO2 - MPI/OpenMP Logic] TSP sequencing completed in "
              << orderMs << " ms." << std::endl;

    // --- STEP 3: Setup Video Writer ---
    // FourCC code for MP4 (H.264 / MP4V)
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    double fps = 60.0;
    cv::VideoWriter videoWriter(outputVideoPath, fourcc, fps, cv::Size(targetWidth, targetHeight), true);

    if (!videoWriter.isOpened()) {
        std::cerr << "[ERROR] Could not open OpenCV VideoWriter for output '" << outputVideoPath << "'." << std::endl;
        return -1;
    }

    // --- STEP 4: CUDA Accelerated Transition Generation & Video Encoding ---
    unsigned char *d_imgA, *d_imgB, *d_output;
    cudaMalloc(&d_imgA, imgSize);
    cudaMalloc(&d_imgB, imgSize);
    cudaMalloc(&d_output, imgSize);

    // Host frame buffer for transfer back from GPU
    cv::Mat h_outputFrame(targetHeight, targetWidth, CV_8UC3);

    long long totalFramesWritten = 0;

    std::cout << "[INFO] [CO3/CO4 - CUDA] Executing GPU kernels and streaming directly into video writer..." << std::endl;
    auto t5 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i + 1 < orderedIndices.size(); ++i) {
        int idxA = orderedIndices[i];
        int idxB = orderedIndices[i+1];

        // Transfer current image pair to GPU
        cudaMemcpy(d_imgA, preloadedImages[idxA].data, imgSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_imgB, preloadedImages[idxB].data, imgSize, cudaMemcpyHostToDevice);

        // Write base image A
        videoWriter.write(preloadedImages[idxA]);
        totalFramesWritten++;

        // Generate and write intermediate transition frames on GPU
        for (int f = 1; f <= intermediateFrames; ++f) {
            float alpha = (float)f / (intermediateFrames + 1);
            launchBlendKernel(d_imgA, d_imgB, d_output, alpha, targetWidth, targetHeight);

            // Copy back to host frame buffer
            cudaMemcpy(h_outputFrame.data, d_output, imgSize, cudaMemcpyDeviceToHost);

            // Write directly to video stream
            videoWriter.write(h_outputFrame);
            totalFramesWritten++;
        }
    }

    // Write final image
    videoWriter.write(preloadedImages[orderedIndices.back()]);
    totalFramesWritten++;

    cudaDeviceSynchronize();
    videoWriter.release(); // Close and save video file

    auto t6 = std::chrono::high_resolution_clock::now();
    const double totalDuration = std::chrono::duration<double, std::milli>(t6 - t5).count();

    std::cout << "[SUCCESS] Generated and encoded " << totalFramesWritten
              << " frames into '" << outputVideoPath << "' in " << totalDuration << " ms ("
              << (totalDuration / totalFramesWritten) << " ms/frame)." << std::endl;

    if (!writeMetrics(opt, "parallel", numImages, featureMs, orderMs, totalDuration, totalFramesWritten))
        return -1;

    // Cleanup GPU Memory
    cudaFree(d_imgA);
    cudaFree(d_imgB);
    cudaFree(d_output);

    std::cout << "[SUCCESS] Project Complete! All Course Outcomes Satisfied." << std::endl;
    return 0;
}
