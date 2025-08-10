// src/01_simple_trt_test.cpp
//
// Purpose: A minimal C++ program to verify the TensorRT installation.
// It initializes the TensorRT builder and prints the library version.

#include <iostream>
#include <memory>
#include "NvInfer.h"
#include <trt_utils/trt_common.h>

int main(int argc, char** argv) {
    TrtLogger gLogger;

    std::cout << "Attempting to create a TensorRT builder..." << std::endl;

    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(gLogger));

    if (!builder) {
        std::cerr << "Error: Failed to create the TensorRT builder." << std::endl;
        return -1;
    }

    std::cout << "TensorRT builder created successfully." << std::endl;

    std::cout << "TensorRT Version (compile time): "
              << NV_TENSORRT_MAJOR << "."
              << NV_TENSORRT_MINOR << "."
              << NV_TENSORRT_PATCH << std::endl;

    std::cout << "TensorRT builder will be destroyed automatically." << std::endl;

    return 0;
}
