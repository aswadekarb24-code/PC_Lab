#include <iostream>
#include <vector>
#include <string>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "common.hpp"
#include "blocking_queue.hpp"

// Declare CUDA host wrapper function defined in kernels.cu
extern "C" cudaError_t launchBlendKernel(const unsigned char* d_imgA, const unsigned char* d_imgB,
                                         unsigned char* d_output, float alpha, int width, int height,
                                         cudaStream_t stream);

// One frame handed from the GPU producer to the encoder thread, in video order.
// slot >= 0: a blended frame in pinned buffer `slot`; slot < 0: `direct` is an original image.
struct FrameItem {
    int slot = -1;
    const cv::Mat* direct = nullptr;
};


constexpr int kNumSlots = 8;  // frames in flight per segment

// Single-quote a string for /bin/sh.
static std::string shellQuote(const std::string& s) {
    std::string r = "'";
    for (char c : s) r += (c == '\'') ? std::string("'\\''") : std::string(1, c);
    return r + "'";
}

// Encodes the image pairs [pairBegin, pairEnd) of the ordered sequence (and the final image if
// includeLast) into its own video file. The calling thread is the producer: it uploads each pair and
// queues CUDA blend kernels + async device-to-host copies on per-slot streams. A helper thread is the
// consumer: it pops frames in order and feeds this segment's VideoWriter, so blending overlaps encoding.
// Several segments run concurrently, each with its own writer, buffers and streams.
static bool encodeSegment(const std::vector<cv::Mat>& images, const std::vector<int>& order,
                          size_t pairBegin, size_t pairEnd, bool includeLast,
                          const std::string& outPath, int width, int height, int intermediateFrames,
                          long long& framesWritten) {
    const size_t imgSize = (size_t)width * height * 3;
    cv::VideoWriter videoWriter(outPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 60.0,
                                cv::Size(width, height), true);
    if (!videoWriter.isOpened()) {
        std::cerr << "[ERROR] Could not open OpenCV VideoWriter for output '" << outPath << "'." << std::endl;
        return false;
    }

    std::atomic<bool> failed{false};
    BlockingQueue<int> freeSlots;               // slot ids the producer may fill
    BlockingQueue<FrameItem> readyFrames;       // frames in video order for the encoder
    auto abortPipeline = [&](const char* what, cudaError_t err) {
        if (!failed.exchange(true))
            std::cerr << "[ERROR] " << what << ": " << cudaGetErrorString(err) << std::endl;
        freeSlots.close();
        readyFrames.close();
    };

    // Two device image-pair buffers (double-buffered across pairs), per-slot output buffers.
    unsigned char* d_img[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
    unsigned char* d_out[kNumSlots] = {};
    unsigned char* h_out[kNumSlots] = {};       // pinned host memory
    cudaStream_t slotStream[kNumSlots] = {};
    cudaStream_t uploadStream = nullptr;
    cudaEvent_t pairDone[2][kNumSlots] = {};    // last use of d_img[b] on each slot stream

    cudaError_t err = cudaSuccess;
    for (int b = 0; b < 2 && err == cudaSuccess; ++b)
        for (int k = 0; k < 2 && err == cudaSuccess; ++k) err = cudaMalloc(&d_img[b][k], imgSize);
    if (err == cudaSuccess) err = cudaStreamCreateWithFlags(&uploadStream, cudaStreamNonBlocking);
    for (int s = 0; s < kNumSlots && err == cudaSuccess; ++s) {
        err = cudaMalloc(&d_out[s], imgSize);
        if (err == cudaSuccess) err = cudaMallocHost(&h_out[s], imgSize);
        if (err == cudaSuccess) err = cudaStreamCreateWithFlags(&slotStream[s], cudaStreamNonBlocking);
        for (int b = 0; b < 2 && err == cudaSuccess; ++b)
            err = cudaEventCreateWithFlags(&pairDone[b][s], cudaEventDisableTiming);
    }
    if (err != cudaSuccess) abortPipeline("CUDA setup failed", err);
    else for (int s = 0; s < kNumSlots; ++s) freeSlots.push(s);

    framesWritten = 0;
    std::thread consumer;
    if (!failed) {
        consumer = std::thread([&] {
            FrameItem item;
            while (readyFrames.pop(item)) {
                if (item.slot < 0) {
                    videoWriter.write(*item.direct);
                } else {
                    cudaError_t e = cudaStreamSynchronize(slotStream[item.slot]);
                    if (e != cudaSuccess) { abortPipeline("CUDA stream sync failed", e); break; }
                    videoWriter.write(cv::Mat(height, width, CV_8UC3, h_out[item.slot]));
                    freeSlots.push(item.slot);
                }
                ++framesWritten;
            }
        });

        for (size_t i = pairBegin; i < pairEnd && !failed; ++i) {
            const cv::Mat& imgA = images[order[i]];
            const cv::Mat& imgB = images[order[i + 1]];
            const int b = (int)(i & 1);

            if (intermediateFrames > 0) {
                // d_img[b] was last used by pair i-2: wait for those kernels before overwriting.
                for (int s = 0; s < kNumSlots && err == cudaSuccess; ++s)
                    err = cudaEventSynchronize(pairDone[b][s]);
                if (err == cudaSuccess) err = cudaMemcpyAsync(d_img[b][0], imgA.data, imgSize, cudaMemcpyHostToDevice, uploadStream);
                if (err == cudaSuccess) err = cudaMemcpyAsync(d_img[b][1], imgB.data, imgSize, cudaMemcpyHostToDevice, uploadStream);
                if (err == cudaSuccess) err = cudaStreamSynchronize(uploadStream);
                if (err != cudaSuccess) { abortPipeline("CUDA image upload failed", err); break; }
            }

            readyFrames.push({-1, &imgA});

            for (int f = 1; f <= intermediateFrames; ++f) {
                int slot;
                if (!freeSlots.pop(slot)) break;   // pipeline aborted
                float alpha = (float)f / (intermediateFrames + 1);
                err = launchBlendKernel(d_img[b][0], d_img[b][1], d_out[slot], alpha,
                                        width, height, slotStream[slot]);
                if (err == cudaSuccess)
                    err = cudaMemcpyAsync(h_out[slot], d_out[slot], imgSize,
                                          cudaMemcpyDeviceToHost, slotStream[slot]);
                if (err != cudaSuccess) { abortPipeline("CUDA blend launch failed", err); break; }
                readyFrames.push({slot, nullptr});
            }

            if (intermediateFrames > 0 && !failed)
                for (int s = 0; s < kNumSlots; ++s) cudaEventRecord(pairDone[b][s], slotStream[s]);
        }

        // Final image (last segment only), then let the consumer drain the queue and finish encoding.
        if (!failed && includeLast) readyFrames.push({-1, &images[order.back()]});
        readyFrames.close();
        consumer.join();
    }

    videoWriter.release();

    cudaDeviceSynchronize();
    for (int b = 0; b < 2; ++b) {
        for (int k = 0; k < 2; ++k) cudaFree(d_img[b][k]);
        for (int s = 0; s < kNumSlots; ++s) if (pairDone[b][s]) cudaEventDestroy(pairDone[b][s]);
    }
    for (int s = 0; s < kNumSlots; ++s) {
        cudaFree(d_out[s]);
        cudaFreeHost(h_out[s]);
        if (slotStream[s]) cudaStreamDestroy(slotStream[s]);
    }
    if (uploadStream) cudaStreamDestroy(uploadStream);
    return !failed;
}

int main(int argc, char** argv) {
    Options opt;
    bool showHelp = false;
    if (!parseArgs(argc, argv, opt, showHelp)) return 2;
    if (showHelp) { printUsage(argv[0]); return 0; }
    if (opt.threads > 0) omp_set_num_threads(opt.threads);
    opt.threads = omp_get_max_threads();
    applyCvThreads(opt);

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
    std::cout << "[INFO] OpenMP threads: " << opt.threads << ", OpenCV threads: " << opt.cvThreads << "\n";
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

    // --- STEP 3/4: Segmented CUDA blending + video encoding ---
    // The ordered sequence is split into `segments` contiguous ranges of image pairs. Each range is
    // blended and encoded concurrently (GPU producer -> queue -> encoder thread, see encodeSegment) into
    // its own file; the files are then joined without re-encoding. segments == 1 writes one file directly.
    const size_t numPairs = orderedIndices.size() - 1;
    const size_t K = std::min<size_t>((size_t)opt.segments, numPairs);
    if (K > 1 && std::system("ffmpeg -version >/dev/null 2>&1") != 0) {
        std::cerr << "[ERROR] --segments > 1 needs the 'ffmpeg' executable in PATH to join the segments." << std::endl;
        return -1;
    }

    const fs::path outPath(outputVideoPath);
    auto segmentPath = [&](size_t j) {
        if (K == 1) return outputVideoPath;
        return (outPath.parent_path() / (outPath.stem().string() + ".seg" + std::to_string(j) +
                (outPath.extension().empty() ? ".mp4" : outPath.extension().string()))).string();
    };

    std::cout << "[INFO] [CO3/CO4 - CUDA] " << K << " segment(s), each: GPU producer -> queue ("
              << kNumSlots << " slots) -> encoder thread..." << std::endl;
    auto t5 = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> workers;
    std::vector<char> segOk(K, 0);
    std::vector<long long> segFrames(K, 0);
    for (size_t j = 0; j < K; ++j) {
        const size_t pb = j * numPairs / K, pe = (j + 1) * numPairs / K;
        workers.emplace_back([&, j, pb, pe] {
            segOk[j] = encodeSegment(preloadedImages, orderedIndices, pb, pe, j + 1 == K, segmentPath(j),
                                     targetWidth, targetHeight, intermediateFrames, segFrames[j]);
        });
    }
    for (auto& w : workers) w.join();

    bool ok = true;
    long long totalFramesWritten = 0;
    for (size_t j = 0; j < K; ++j) { ok = ok && segOk[j]; totalFramesWritten += segFrames[j]; }

    if (ok && K > 1) {
        const std::string listPath = outputVideoPath + ".segments.txt";
        {
            std::ofstream list(listPath);
            for (size_t j = 0; j < K; ++j) {
                std::string p = fs::absolute(segmentPath(j)).string(), esc;
                for (char c : p) esc += (c == '\'') ? std::string("'\\''") : std::string(1, c);
                list << "file '" << esc << "'\n";
            }
        }
        const std::string cmd = "ffmpeg -y -loglevel error -f concat -safe 0 -i " + shellQuote(listPath) +
                                " -c copy " + shellQuote(outputVideoPath);
        if (std::system(cmd.c_str()) != 0) {
            std::cerr << "[ERROR] ffmpeg failed to join the video segments." << std::endl;
            ok = false;
        }
        std::error_code ec;
        fs::remove(listPath, ec);
    }
    if (K > 1) {
        std::error_code ec;
        for (size_t j = 0; j < K; ++j) fs::remove(segmentPath(j), ec);
    }

    auto t6 = std::chrono::high_resolution_clock::now();
    const double totalDuration = std::chrono::duration<double, std::milli>(t6 - t5).count();

    if (!ok) return -1;

    std::cout << "[SUCCESS] Generated and encoded " << totalFramesWritten
              << " frames into '" << outputVideoPath << "' in " << totalDuration << " ms ("
              << (totalDuration / totalFramesWritten) << " ms/frame)." << std::endl;

    if (!writeMetrics(opt, "parallel", numImages, featureMs, orderMs, totalDuration, totalFramesWritten))
        return -1;

    std::cout << "[SUCCESS] Project Complete! All Course Outcomes Satisfied." << std::endl;
    return 0;
}
