#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <omp.h>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// Structure to store extracted features for each image
struct ImageFeature {
    std::string filepath;
    cv::Mat histogram; // 3D Color Histogram in HSV space
};

// Extract 3D HSV Color Histogram (Visual Feature Representation)
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
    std::cout << "[INFO] Found " << numImages << " images in " << datasetPath << std::endl;
    std::cout << "[INFO] Running on OpenMP with max threads: " << omp_get_max_threads() << std::endl;

    std::vector<ImageFeature> features(numImages);

    auto startTime = std::chrono::high_resolution_clock::now();

    // OpenMP Parallel For Loop over CPU cores
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < numImages; ++i) {
        cv::Mat img = cv::imread(imagePaths[i]);
        if (!img.empty()) {
            // Resize for uniform processing (e.g., 720p)
            cv::resize(img, img, cv::Size(1280, 720));
            
            features[i].filepath = imagePaths[i];
            features[i].histogram = extractHSVHistogram(img);
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    std::cout << "[SUCCESS] Processed and extracted features for " << numImages 
              << " images in " << duration << " ms (" 
              << (duration / numImages) << " ms/image)" << std::endl;

    return 0;
}