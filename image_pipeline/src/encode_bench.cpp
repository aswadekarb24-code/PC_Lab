// Encoder scaling benchmark: T fully independent copies of the same encode (own VideoWriter, own output
// file, shared read-only frames), run concurrently. No GPU, no producer/consumer queue, no join.
// Measures how much encoder throughput this machine offers when several encoders run at once.
//
// Usage: encode_bench --input <video> --frames <M> --copies <T> [--output-dir <dir>]
// Prints one line: copies, frames per copy, wall time (ms), aggregate frames per second.
#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    std::string input, outDir = ".";
    int frames = 500, copies = 1;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--input") input = v;
        else if (a == "--frames") frames = std::stoi(v);
        else if (a == "--copies") copies = std::stoi(v);
        else if (a == "--output-dir") outDir = v;
        else { std::cerr << "[ERROR] unknown option " << a << "\n"; return 2; }
    }
    cv::VideoCapture cap(input);
    if (!cap.isOpened()) { std::cerr << "[ERROR] cannot open " << input << "\n"; return 1; }
    std::vector<cv::Mat> clip;
    cv::Mat f;
    while ((int)clip.size() < frames && cap.read(f)) clip.push_back(f.clone());
    if (clip.empty()) { std::cerr << "[ERROR] no frames\n"; return 1; }
    const cv::Size size(clip[0].cols, clip[0].rows);

    std::atomic<bool> ok{true};
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    for (int c = 0; c < copies; ++c) {
        workers.emplace_back([&, c] {
            cv::VideoWriter w(outDir + "/enc_bench_" + std::to_string(c) + ".mp4",
                              cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 60.0, size, true);
            if (!w.isOpened()) { ok = false; return; }
            for (const cv::Mat& fr : clip) w.write(fr);
            w.release();
        });
    }
    for (auto& w : workers) w.join();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) { std::cerr << "[ERROR] a writer failed\n"; return 1; }
    std::cout << copies << " " << clip.size() << " " << ms << " " << (copies * clip.size() * 1000.0 / ms) << "\n";
    return 0;
}
