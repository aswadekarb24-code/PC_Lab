#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

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
    if (numImages == 0) {
        std::cerr << "[ERROR] No images found in " << datasetPath << std::endl;
        return -1;
    }

    std::cout << "[INFO] Found " << numImages << " images." << std::endl;
    std::cout << "[INFO] Maximum OpenMP threads available: " << omp_get_max_threads() << std::endl;

    // --- STEP 1: Parallel Feature Extraction ---
    std::vector<cv::Mat> histograms(numImages);
    auto t1 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < numImages; ++i) {
        cv::Mat img = cv::imread(imagePaths[i]);
        if (!img.empty()) {
            cv::resize(img, img, cv::Size(640, 360)); // Lower resolution for fast feature comparison
            histograms[i] = extractHSVHistogram(img);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "[SUCCESS] Extracted histograms in " 
              << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms." << std::endl;

    // --- STEP 2: Parallel NxN Similarity Matrix Computation ---
    // similarityMatrix[i][j] stores correlation score between image i and image j (range: 0 to 1)
    std::vector<std::vector<float>> similarityMatrix(numImages, std::vector<float>(numImages, 0.0f));

    auto t3 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < numImages; ++i) {
        for (size_t j = 0; j < numImages; ++j) {
            if (i == j) {
                similarityMatrix[i][j] = 1.0f;
            } else if (j > i) {
                // cv::HISTCMP_CORREL returns 1.0 for identical, -1.0 for completely opposite
                float score = (float)cv::compareHist(histograms[i], histograms[j], cv::HISTCMP_CORREL);
                similarityMatrix[i][j] = score;
                similarityMatrix[j][i] = score; // Symmetric matrix
            }
        }
    }

    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "[SUCCESS] Computed " << (numImages * numImages) 
              << " pairwise comparisons in " 
              << std::chrono::duration<double, std::milli>(t4 - t3).count() << " ms." << std::endl;

    // --- STEP 3: Greedy Nearest Neighbor (TSP) Image Sequencing ---
    auto t5 = std::chrono::high_resolution_clock::now();

    std::vector<int> orderedIndices;
    orderedIndices.reserve(numImages);
    std::vector<bool> visited(numImages, false);

    // Start with the first image
    int currentIndex = 0;
    visited[currentIndex] = true;
    orderedIndices.push_back(currentIndex);

    for (size_t step = 1; step < numImages; ++step) {
        int nextIndex = -1;
        float maxSimilarity = -2.0f;

        // Find the unvisited image with the highest similarity score to current image
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

    auto t6 = std::chrono::high_resolution_clock::now();
    std::cout << "[SUCCESS] Generated optimal image sequence of length " << orderedIndices.size() 
              << " in " << std::chrono::duration<double, std::milli>(t6 - t5).count() << " ms." << std::endl;

    // Print first 10 sorted sequence paths as verification
    std::cout << "\n[PREVIEW] First 10 sorted image sequence paths:\n";
    for (int i = 0; i < 10 && i < numImages; ++i) {
        std::cout << "  " << i + 1 << ". " << imagePaths[orderedIndices[i]] << "\n";
    }

    return 0;
}