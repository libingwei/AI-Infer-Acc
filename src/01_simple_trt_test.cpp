// src/01_simple_trt_test.cpp
//
// Purpose: A minimal C++ program to verify the TensorRT installation.
// It initializes the TensorRT builder and prints the library version.
// This is the "hello world" of TensorRT C++ programming.

#include <iostream>
#include "NvInfer.h"

// Helper for using smart pointers with TensorRT objects
template <typename T>
using UniquePtr = std::unique_ptr<T, void (*)(T*)>;

// A simple logger class required by the TensorRT API.
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress info-level messages
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

int main(int argc, char** argv) {
    Logger gLogger;

    std::cout << "Attempting to create a TensorRT builder..." << std::endl;

    // Create the core IBuilder object using a smart pointer for automatic memory management.
    UniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(gLogger), [](nvinfer1::IBuilder* b) { b->destroy(); });

    if (!builder) {
        std::cerr << "Error: Failed to create the TensorRT builder." << std::endl;
        return -1;
    }

    std::cout << "TensorRT builder created successfully." << std::endl;

    // Print the version of TensorRT that this program was compiled against.
    std::cout << "TensorRT Version (compile time): "
              << NV_TENSORRT_MAJOR << "."
              << NV_TENSORRT_MINOR << "."
              << NV_TENSORRT_PATCH << std::endl;

    // No need to manually call destroy(); the UniquePtr will handle it when it goes out of scope.
    std::cout << "TensorRT builder will be destroyed automatically." << std::endl;

    return 0;
}
