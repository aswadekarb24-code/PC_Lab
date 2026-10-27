#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "common.hpp"

// CPU equivalent of blendKernel in kernels.cu (same formula and truncation)
static void blendFrameCPU(const cv::Mat& a, const cv::Mat& b, cv::Mat& out, float alpha) {
    const float invAlpha = 1.0f - alpha;
    const size_t total = (size_t)a.total() * a.channels();
    const unsigned char* pa = a.data;
    const unsigned char* pb = b.data;
    unsigned char* po = out.data;
    for (size_t k = 0; k < total; ++k)
        po[k] = (unsigned char)(pa[k] * invAlpha + pb[k] * alpha);
}

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

    std::cout << "[INFO] Mode: serial (single-threaded CPU)\n";
    std::cout << "[INFO] Found " << imagePaths.size() << " images. Resolution "
              << targetWidth << "x" << targetHeight << ", " << intermediateFrames
              << " intermediate frames.\n";

    // --- STEP 1: Serial Feature Extraction ---
    size_t numImages = imagePaths.size();
    std::vector<cv::Mat> histograms(numImages);
    std::vector<cv::Mat> preloadedImages(numImages);
    auto t1 = std::chrono::high_resolution_clock::now();

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
    std::cout << "[SUCCESS] [Serial] Feature extraction completed in "
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

    // --- STEP 2: Serial NxN Similarity Matrix & Greedy Sorting ---
    auto t3 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> similarityMatrix(numImages, std::vector<float>(numImages, 0.0f));

    for (size_t i = 0; i < numImages; ++i) {
        for (size_t j = i + 1; j < numImages; ++j) {
            float score = (float)cv::compareHist(histograms[i], histograms[j], cv::HISTCMP_CORREL);
            similarityMatrix[i][j] = score;
            similarityMatrix[j][i] = score;
        }
    }

    std::vector<int> orderedIndices = greedyOrder(similarityMatrix);
    auto t4 = std::chrono::high_resolution_clock::now();
    const double orderMs = std::chrono::duration<double, std::milli>(t4 - t3).count();
    std::cout << "[SUCCESS] [Serial] TSP sequencing completed in "
              << orderMs << " ms." << std::endl;

    // --- STEP 3: Setup Video Writer ---
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    double fps = 60.0;
    cv::VideoWriter videoWriter(outputVideoPath, fourcc, fps, cv::Size(targetWidth, targetHeight), true);

    if (!videoWriter.isOpened()) {
        std::cerr << "[ERROR] Could not open OpenCV VideoWriter for output '" << outputVideoPath << "'." << std::endl;
        return -1;
    }

    // --- STEP 4: CPU Transition Generation & Video Encoding ---
    cv::Mat outputFrame(targetHeight, targetWidth, CV_8UC3);
    long long totalFramesWritten = 0;

    std::cout << "[INFO] [Serial] Blending frames on CPU and streaming directly into video writer..." << std::endl;
    auto t5 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i + 1 < orderedIndices.size(); ++i) {
        const cv::Mat& imgA = preloadedImages[orderedIndices[i]];
        const cv::Mat& imgB = preloadedImages[orderedIndices[i+1]];

        videoWriter.write(imgA);
        totalFramesWritten++;

        for (int f = 1; f <= intermediateFrames; ++f) {
            float alpha = (float)f / (intermediateFrames + 1);
            blendFrameCPU(imgA, imgB, outputFrame, alpha);
            videoWriter.write(outputFrame);
            totalFramesWritten++;
        }
    }

    videoWriter.write(preloadedImages[orderedIndices.back()]);
    totalFramesWritten++;

    videoWriter.release();

    auto t6 = std::chrono::high_resolution_clock::now();
    const double totalDuration = std::chrono::duration<double, std::milli>(t6 - t5).count();

    std::cout << "[SUCCESS] Generated and encoded " << totalFramesWritten
              << " frames into '" << outputVideoPath << "' in " << totalDuration << " ms ("
              << (totalDuration / totalFramesWritten) << " ms/frame)." << std::endl;

    if (!writeMetrics(opt, "serial", numImages, featureMs, orderMs, totalDuration, totalFramesWritten))
        return -1;

    std::cout << "[SUCCESS] Serial pipeline complete!" << std::endl;
    return 0;
}
