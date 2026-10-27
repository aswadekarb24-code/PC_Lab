#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

struct Options {
    std::string datasetPath = "../data/raw_images";
    int width = 1280;
    int height = 720;
    int intermediateFrames = 30;
    std::string outputPath = "output_transition_video.mp4";
    std::string metricsPath;  // optional: JSON file with the timings
};

inline void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --dataset <path>   Image folder (default: ../data/raw_images)\n"
              << "  --width <int>      Output width  (default: 1280; requires --height)\n"
              << "  --height <int>     Output height (default: 720; requires --width)\n"
              << "  --frames <int>     Intermediate frames per image pair (default: 30)\n"
              << "  --output <path>    Output video path (default: output_transition_video.mp4)\n"
              << "  --metrics <path>   Write timings (ms) as JSON to this file (optional)\n"
              << "  --help             Show this message\n";
}

inline bool parsePositiveInt(const std::string& s, int minValue, int& out) {
    try {
        size_t pos = 0;
        long v = std::stol(s, &pos);
        if (pos != s.size() || v < minValue || v > 100000) return false;
        out = (int)v;
        return true;
    } catch (...) {
        return false;
    }
}

// Returns false on error (message already printed). Sets showHelp if --help was given.
inline bool parseArgs(int argc, char** argv, Options& opt, bool& showHelp) {
    showHelp = false;
    bool haveW = false, haveH = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { showHelp = true; return true; }

        if (a != "--dataset" && a != "--width" && a != "--height" &&
            a != "--frames" && a != "--output" && a != "--metrics") {
            std::cerr << "[ERROR] Unknown or positional argument '" << a << "'. Use --help.\n";
            return false;
        }
        if (i + 1 >= argc) {
            std::cerr << "[ERROR] Missing value for " << a << ".\n";
            return false;
        }
        std::string v = argv[++i];

        if (a == "--dataset") opt.datasetPath = v;
        else if (a == "--output") opt.outputPath = v;
        else if (a == "--metrics") opt.metricsPath = v;
        else if (a == "--width") {
            if (!parsePositiveInt(v, 1, opt.width)) {
                std::cerr << "[ERROR] --width must be a positive integer, got '" << v << "'.\n";
                return false;
            }
            haveW = true;
        } else if (a == "--height") {
            if (!parsePositiveInt(v, 1, opt.height)) {
                std::cerr << "[ERROR] --height must be a positive integer, got '" << v << "'.\n";
                return false;
            }
            haveH = true;
        } else if (a == "--frames") {
            if (!parsePositiveInt(v, 0, opt.intermediateFrames)) {
                std::cerr << "[ERROR] --frames must be a non-negative integer, got '" << v << "'.\n";
                return false;
            }
        }
    }
    if (haveW != haveH) {
        std::cerr << "[ERROR] --width and --height must be given together (or neither).\n";
        return false;
    }
    return true;
}

// Writes the three stage timings (ms) to opt.metricsPath if one was requested.
inline bool writeMetrics(const Options& opt, const char* mode, size_t numImages,
                         double featureMs, double orderMs, double encodeMs, long long frames) {
    if (opt.metricsPath.empty()) return true;
    std::ofstream f(opt.metricsPath);
    if (!f) {
        std::cerr << "[ERROR] Cannot write metrics file '" << opt.metricsPath << "'.\n";
        return false;
    }
    f << "{\n"
      << "  \"mode\": \"" << mode << "\",\n"
      << "  \"num_images\": " << numImages << ",\n"
      << "  \"width\": " << opt.width << ",\n"
      << "  \"height\": " << opt.height << ",\n"
      << "  \"intermediate_frames\": " << opt.intermediateFrames << ",\n"
      << "  \"total_frames\": " << frames << ",\n"
      << "  \"feature_extraction_ms\": " << featureMs << ",\n"
      << "  \"sequencing_ms\": " << orderMs << ",\n"
      << "  \"encoding_ms\": " << encodeMs << "\n"
      << "}\n";
    return true;
}

inline std::vector<std::string> listImages(const std::string& datasetPath) {
    std::vector<std::string> paths;
    try {
        for (const auto& entry : fs::directory_iterator(datasetPath)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return (char)std::tolower(c); });
            if (ext == ".jpg" || ext == ".png" || ext == ".jpeg")
                paths.push_back(entry.path().string());
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "[ERROR] Cannot read dataset path '" << datasetPath << "': " << e.what() << "\n";
        paths.clear();
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

// Extract 3D HSV Color Histogram
inline cv::Mat extractHSVHistogram(const cv::Mat& img) {
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

// Greedy nearest-neighbour ordering starting at index 0.
inline std::vector<int> greedyOrder(const std::vector<std::vector<float>>& sim) {
    size_t n = sim.size();
    std::vector<int> ordered;
    ordered.reserve(n);
    std::vector<bool> visited(n, false);

    int current = 0;
    visited[current] = true;
    ordered.push_back(current);

    for (size_t step = 1; step < n; ++step) {
        int next = -1;
        float best = -2.0f;
        for (size_t j = 0; j < n; ++j) {
            if (!visited[j] && sim[current][j] > best) {
                best = sim[current][j];
                next = (int)j;
            }
        }
        if (next == -1) break;
        visited[next] = true;
        ordered.push_back(next);
        current = next;
    }
    return ordered;
}
