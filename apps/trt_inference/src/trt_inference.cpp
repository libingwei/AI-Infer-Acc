#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <numeric>
#include <string>
#include <chrono>
#include <cstdlib>

#include <trt_utils/trt_runtime.h>

int main(int argc, char** argv) {
    if (argc < 2 || argc > 6) {
        std::cerr << "Usage: " << argv[0] << " <path_to_engine_file> [batch_size] [--use-default-stream] [--pinned] [--pipeline]" << std::endl;
        std::cerr << "  If --use-default-stream is omitted, a non-default CUDA stream will be created and used." << std::endl;
        std::cerr << "  Use --pinned or env USE_PINNED=1 to allocate pinned host memory for H2D/D2H." << std::endl;
        std::cerr << "  Use --pipeline to enable double-buffered copy/compute overlap (uses two streams)." << std::endl;
        return -1;
    }
    const char* engine_filename = argv[1];
    int batch_size = 1;
    bool useDefaultStream = false;
    bool usePinned = false;
    bool usePipeline = false;
    if (argc >= 3) {
        // argv[2] can be batch_size or the flag
        std::string a2 = argv[2];
        if (a2 == "--use-default-stream") {
            useDefaultStream = true;
        } else {
            batch_size = std::stoi(a2);
        }
    }
    if (argc >= 4) {
        for (int i = 3; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--use-default-stream") useDefaultStream = true;
            if (a == "--pinned") usePinned = true;
            if (a == "--pipeline") usePipeline = true;
        }
    }

    // Also allow env override
    if (const char* env = std::getenv("USE_DEFAULT_STREAM")) {
        if (std::string(env) == "1" || std::string(env) == "true") useDefaultStream = true;
    }
    if (const char* envp = std::getenv("USE_PINNED")) {
        if (std::string(envp) == "1" || std::string(envp) == "true") usePinned = true;
    }

    // Detect precision mode from filename
    std::string filename_str(engine_filename);
    if (filename_str.find("_fp16") != std::string::npos) {
        std::cout << "Running inference in FP16 mode." << std::endl;
    } else if (filename_str.find("_int8") != std::string::npos) {
        std::cout << "Running inference in INT8 mode." << std::endl;
    } else {
        std::cout << "Running inference in FP32 mode." << std::endl;
    }
    std::cout << "Using Batch Size: " << batch_size << std::endl;


    TrtLogger gLogger;
    TrtRunner runner(gLogger);
    if (!runner.loadEngineFromFile(engine_filename)) { std::cerr << "Failed to load engine" << std::endl; return -1; }
    if (!runner.prepare(batch_size)) { std::cerr << "Failed to prepare context" << std::endl; return -1; }

    // 5. Prepare and copy input data
    std::vector<float> host_input_vec; float* host_input = nullptr; // raw pointer for pinned path
    std::vector<float> host_output_vec; float* host_output = nullptr;
    size_t input_size = runner.inputSize();
    size_t output_size = runner.outputSize();
    if (usePinned) {
        CHECK(cudaHostAlloc((void**)&host_input, input_size * sizeof(float), cudaHostAllocDefault));
        CHECK(cudaHostAlloc((void**)&host_output, output_size * sizeof(float), cudaHostAllocDefault));
        for (size_t i = 0; i < input_size; ++i) host_input[i] = 1.0f;
        std::cout << "Using pinned host memory for H2D/D2H." << std::endl;
    } else {
        host_input_vec.assign(input_size, 1.0f);
        host_output_vec.resize(output_size);
        host_input = host_input_vec.data();
        host_output = host_output_vec.data();
    }
    // --- Performance Test ---
    int num_iterations = 100;
    std::cout << "\n--- Running Performance Test ---" << std::endl;
    std::cout << "Number of iterations: " << num_iterations << std::endl;
    // Timed runs via runner
    float total_duration_ms = runner.run(num_iterations, host_input, host_output, useDefaultStream, usePinned, usePipeline);
    double avg_latency_ms = total_duration_ms / num_iterations;
    double throughput_fps = (batch_size * num_iterations) / (total_duration_ms / 1000.0);

    std::cout << "\n--- Performance Results ---" << std::endl;
    std::cout << "Total time for " << num_iterations << " inferences: " << total_duration_ms << " ms" << std::endl;
    std::cout << "Average Latency: " << avg_latency_ms << " ms" << std::endl;
    std::cout << "Throughput: " << throughput_fps << " FPS (frames per second)" << std::endl;
    std::cout << "---------------------------\n" << std::endl;

    // Print first 10 results to confirm it's still working
    std::cout << "Displaying first 10 output values from the last run:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << host_output[i] << " ";
    }
    std::cout << std::endl;

    if (usePinned) {
        CHECK(cudaFreeHost(host_input));
        CHECK(cudaFreeHost(host_output));
    }

    return 0;
}
