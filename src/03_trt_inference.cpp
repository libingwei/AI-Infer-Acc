#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <numeric>
#include <string>
#include <chrono> // For timing

#include "NvInfer.h"
#include "cuda_runtime_api.h"

// A simple logger class required by the TensorRT API.
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

// Helper macro to check for CUDA errors
#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            std::cerr << "Cuda failure: " << cudaGetErrorString(ret) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            abort(); \
        } \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_engine_file> [batch_size]" << std::endl;
        return -1;
    }
    const char* engine_filename = argv[1];
    int batch_size = 1;
    if (argc == 3) {
        batch_size = std::stoi(argv[2]);
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


    Logger gLogger;

    // 1. Read the engine file
    std::ifstream engine_file(engine_filename, std::ios::binary);
    if (!engine_file) {
        std::cerr << "Error opening engine file: " << engine_filename << std::endl;
        return -1;
    }
    engine_file.seekg(0, std::ios::end);
    size_t engine_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(engine_size);
    engine_file.read(engine_data.data(), engine_size);

    // 2. Create runtime, engine, and context
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(gLogger));
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(engine_data.data(), engine_size));
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());

    // 3. Set the input dimensions for this inference execution
    const char* input_name = "input";
    auto input_dims = engine->getTensorShape(input_name);
    input_dims.d[0] = batch_size; // Set the batch size from command line
    if (!context->setInputShape(input_name, input_dims)) {
        std::cerr << "Failed to set input shape. Ensure batch size " << batch_size 
                  << " is within the optimization profile range." << std::endl;
        return -1;
    }

    // 4. Allocate GPU buffers
    void* input_buffer = nullptr;
    void* output_buffer = nullptr;
    const char* output_name = "output";

    auto final_input_dims = context->getTensorShape(input_name);
    size_t input_size = std::accumulate(final_input_dims.d, final_input_dims.d + final_input_dims.nbDims, 1, std::multiplies<int64_t>());
    auto final_output_dims = context->getTensorShape(output_name);
    size_t output_size = std::accumulate(final_output_dims.d, final_output_dims.d + final_output_dims.nbDims, 1, std::multiplies<int64_t>());

    CHECK(cudaMalloc(&input_buffer, input_size * sizeof(float)));
    CHECK(cudaMalloc(&output_buffer, output_size * sizeof(float)));

    context->setTensorAddress(input_name, input_buffer);
    context->setTensorAddress(output_name, output_buffer);

    // 5. Prepare and copy input data
    std::vector<float> host_input(input_size, 1.0f);
    CHECK(cudaMemcpy(input_buffer, host_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));

    // --- Performance Test ---
    int num_iterations = 100;
    std::cout << "\n--- Running Performance Test ---" << std::endl;
    std::cout << "Number of iterations: " << num_iterations << std::endl;

    // Warm-up runs
    for (int i = 0; i < 10; ++i) {
        context->enqueueV3(0);
    }
    CHECK(cudaDeviceSynchronize()); // Wait for warm-up to finish

    // Timed runs
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        context->enqueueV3(0);
    }
    CHECK(cudaDeviceSynchronize()); // Wait for all inferences to finish
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate and print metrics
    auto total_duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double avg_latency_ms = total_duration_ms / num_iterations;
    double throughput_fps = (batch_size * num_iterations) / (total_duration_ms / 1000.0);

    std::cout << "\n--- Performance Results ---" << std::endl;
    std::cout << "Total time for " << num_iterations << " inferences: " << total_duration_ms << " ms" << std::endl;
    std::cout << "Average Latency: " << avg_latency_ms << " ms" << std::endl;
    std::cout << "Throughput: " << throughput_fps << " FPS (frames per second)" << std::endl;
    std::cout << "---------------------------\n" << std::endl;


    // Copy final output data back to host for a quick check
    std::vector<float> host_output(output_size);
    CHECK(cudaMemcpy(host_output.data(), output_buffer, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Print first 10 results to confirm it's still working
    std::cout << "Displaying first 10 output values from the last run:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << host_output[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    CHECK(cudaFree(input_buffer));
    CHECK(cudaFree(output_buffer));

    return 0;
}
