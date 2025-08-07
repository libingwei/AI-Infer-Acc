// src/01_simple_trt_test.cpp
//
// Purpose: A minimal C++ program to verify the TensorRT installation.
// It initializes the TensorRT builder and prints the library version.
// This is the "hello world" of TensorRT C++ programming.

#include <iostream>
#include "NvInfer.h"

// A simple logger class required by the TensorRT API.
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

int main(int argc, char** argv) {
    Logger gLogger;

    std::cout << "Attempting to create a TensorRT builder..." << std::endl;

    // Create the core IBuilder object
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);

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

    // Clean up and release the builder object.
    builder->destroy();
    std::cout << "TensorRT builder destroyed." << std::endl;

    return 0;
}
