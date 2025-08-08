// src/03_trt_inference.cpp
//
// Purpose: Loads a pre-built TensorRT engine and performs inference.
// This simulates the actual deployment scenario where the application
// uses the optimized engine for fast predictions.

#include <iostream>
#include <fstream>
#include <vector>
#include <memory> // Required for std::unique_ptr
#include <numeric> // For std::accumulate

#include "NvInfer.h"
#include "cuda_runtime_api.h"

// A simple logger class required by the TensorRT API.
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress info-level messages for a cleaner output.
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
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_engine_file>" << std::endl;
        return -1;
    }
    const char* engine_filename = argv[1];

    Logger gLogger;

    // 1. Read the engine file into a buffer
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

    // 2. Create a runtime and deserialize the engine using smart pointers with the default deleter
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(gLogger));
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(engine_data.data(), engine_size));
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());

    // 3. Allocate GPU buffers for input and output
    // The new API uses tensor names. We assume the names are "input" and "output"
    // as defined in the ONNX export script.
    const char* input_name = "input";
    const char* output_name = "output";

    void* input_buffer = nullptr;
    void* output_buffer = nullptr;

    // Calculate buffer sizes using tensor shapes
    auto input_dims = engine->getTensorShape(input_name);
    size_t input_size = std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1, std::multiplies<int64_t>());
    auto output_dims = engine->getTensorShape(output_name);
    size_t output_size = std::accumulate(output_dims.d, output_dims.d + output_dims.nbDims, 1, std::multiplies<int64_t>());

    CHECK(cudaMalloc(&input_buffer, input_size * sizeof(float)));
    CHECK(cudaMalloc(&output_buffer, output_size * sizeof(float)));

    // Set the buffer addresses in the execution context
    context->setTensorAddress(input_name, input_buffer);
    context->setTensorAddress(output_name, output_buffer);

    // 4. Prepare dummy input data on the host (CPU)
    std::vector<float> host_input(input_size);
    for (size_t i = 0; i < input_size; ++i) {
        host_input[i] = 1.0f; // Simple dummy data
    }

    // 5. Copy input data from host to device (GPU)
    CHECK(cudaMemcpy(input_buffer, host_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));

    // 6. Execute the inference using enqueueV3
    std::cout << "Executing inference..." << std::endl;
    if (!context->enqueueV3(0)) { // 0 is for the default CUDA stream
        std::cerr << "Failed to enqueue inference." << std::endl;
        return -1;
    }

    // 7. Copy output data from device to host
    std::vector<float> host_output(output_size);
    CHECK(cudaMemcpy(host_output.data(), output_buffer, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Inference finished." << std::endl;

    // 8. Print the first 10 results
    std::cout << "Displaying first 10 output values:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << host_output[i] << " ";
    }
    std::cout << std::endl;

    // 9. Clean up GPU buffers
    CHECK(cudaFree(input_buffer));
    CHECK(cudaFree(output_buffer));
    // All TensorRT objects are cleaned up automatically by their smart pointers.

    return 0;
}
