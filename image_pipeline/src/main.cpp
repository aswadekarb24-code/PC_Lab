#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
namespace fs = std::filesystem;

// Declare CUDA host wrapper function defined in kernels.cu
extern "C" void launchBlendKernel(const unsigned char* d_imgA, const unsigned char* d_imgB,
                                    unsigned char* d_output, float alpha, int width, int height);

// Extract 3D HSV Color Histogram
cv::Mat extractHSVHistogram(const cv::Mat& img) {
    cv::Mat hsv, hist;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    int h_bins = 16, s_bins = 8, v_bins = 8;
    int histSize[] = {h_bins, s_bins, v_bins};
    float h_ranges[] = {0, 180};
    float s_ranges[] = {0, 256};
    float v_ranges[] = {0, 256};
    const float* ranges[] = {h_ranges, s_ranges, v_ranges};
    int channels[] = {0, 1, 2};
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 3, histSize, ranges, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    return hist;
}

int main(int argc, char** argv) {
    std::string datasetPath = "../data/raw_images";
    if (argc > 1) datasetPath = argv[1];

    std::vector<std::string> imagePaths;
    for (const auto& entry : fs::directory_iterator(datasetPath)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".png" || ext == ".jpeg") {
                imagePaths.push_back(entry.path().string());
            }
        }
    }

    size_t numImages = imagePaths.size();
    if (numImages < 2) {
        std::cerr << "[ERROR] Need at least 2 images." << std::endl;
        return -1;
    }

    std::cout << "[INFO] Found " << numImages << " images. Initializing Hybrid Pipeline...\n";

    int targetWidth = 1280;
    int targetHeight = 720;
    size_t imgSize = targetWidth * targetHeight * 3;

    // --- STEP 1: OpenMP Parallel Feature Extraction ---
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
    std::cout << "[SUCCESS] Feature extraction completed in " 
              << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms." << std::endl;

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
    std::vector<int> orderedIndices;
    orderedIndices.reserve(numImages);
    std::vector<bool> visited(numImages, false);

    int currentIndex = 0;
    visited[currentIndex] = true;
    orderedIndices.push_back(currentIndex);

    for (size_t step = 1; step < numImages; ++step) {
        int nextIndex = -1;
        float maxSimilarity = -2.0f;
        for (size_t j = 0; j < numImages; ++j) {
            if (!visited[j]) {
                float score = similarityMatrix[currentIndex][j];
                if (score > maxSimilarity) {
                    maxSimilarity = score;
                    nextIndex = j;
                }
            }
        }
        if (nextIndex != -1) {
            visited[nextIndex] = true;
            orderedIndices.push_back(nextIndex);
            currentIndex = nextIndex;
        }
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "[SUCCESS] Similarity matrix & TSP sorting completed in " 
              << std::chrono::duration<double, std::milli>(t4 - t3).count() << " ms." << std::endl;

    // --- STEP 3: CUDA Accelerated Transition Generation Across Sorted Sequence ---
    unsigned char *d_imgA, *d_imgB, *d_output;
    cudaMalloc(&d_imgA, imgSize);
    cudaMalloc(&d_imgB, imgSize);
    cudaMalloc(&d_output, imgSize);

    int intermediateFrames = 5; // 5 smooth transition frames between each sorted image pair
    long long totalTransitionFramesGenerated = 0;

    std::cout << "[INFO] Starting CUDA GPU transition generation across sorted image sequence..." << std::endl;
    auto t5 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < orderedIndices.size() - 1; ++i) {
        int idxA = orderedIndices[i];
        int idxB = orderedIndices[i+1];

        // Transfer current image pair to GPU
        cudaMemcpy(d_imgA, preloadedImages[idxA].data, imgSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_imgB, preloadedImages[idxB].data, imgSize, cudaMemcpyHostToDevice);

        // Generate intermediate alpha-blended frames on GPU
        for (int f = 1; f <= intermediateFrames; ++f) {
            float alpha = (float)f / (intermediateFrames + 1);
            launchBlendKernel(d_imgA, d_imgB, d_output, alpha, targetWidth, targetHeight);
            totalTransitionFramesGenerated++;
        }
    }

    cudaDeviceSynchronize();
    auto t6 = std::chrono::high_resolution_clock::now();
    double cudaDuration = std::chrono::duration<double, std::milli>(t6 - t5).count();

    std::cout << "[SUCCESS] Generated " << totalTransitionFramesGenerated 
              << " transition frames on RTX 4050 in " << cudaDuration << " ms ("
              << (cudaDuration / totalTransitionFramesGenerated) << " ms/frame)." << std::endl;

    // Cleanup GPU Memory
    cudaFree(d_imgA);
    cudaFree(d_imgB);
    cudaFree(d_output);

    std::cout << "[SUCCESS] Complete Hybrid Pipeline Executed Successfully!" << std::endl;
    return 0;
}