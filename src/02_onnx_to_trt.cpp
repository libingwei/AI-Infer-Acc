// src/02_onnx_to_trt.cpp
//
// Purpose: Converts a given ONNX model file into a serialized TensorRT engine.
// This is a crucial step for deployment, as the engine is highly optimized
// for the specific GPU hardware it's built on.

#include <iostream>
#include <fstream>
#include <memory> // Required for std::unique_ptr

#include "NvInfer.h"
#include "NvOnnxParser.h"

// A simple logger class required by the TensorRT API.
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress info-level messages for a cleaner output.
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_onnx_model> <output_engine_path>" << std::endl;
        return -1;
    }
    const char* onnx_filename = argv[1];
    const char* engine_filename = argv[2];

    Logger gLogger;

    // 1. Create core TensorRT objects using smart pointers with the default deleter
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(gLogger));
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, gLogger));

    // 2. Parse the ONNX model file
    std::cout << "Parsing ONNX model: " << onnx_filename << std::endl;
    if (!parser->parseFromFile(onnx_filename, static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        std::cerr << "Failed to parse ONNX file." << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "Parser Error: " << parser->getError(i)->desc() << std::endl;
        }
        return -1;
    }
    std::cout << "ONNX model parsed successfully." << std::endl;

    // 3. Configure the builder
    // Set a memory limit for the workspace. 1GB in this case.
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);

    // 4. Build the serialized engine
    std::cout << "Building TensorRT engine... (This may take a few minutes)" << std::endl;
    std::unique_ptr<nvinfer1::IHostMemory> serialized_engine(builder->buildSerializedNetwork(*network, *config));
    if (!serialized_engine) {
        std::cerr << "Failed to build engine." << std::endl;
        return -1;
    }
    std::cout << "Engine built successfully." << std::endl;

    // 5. Save the engine to a file
    std::ofstream engine_file(engine_filename, std::ios::binary);
    if (!engine_file) {
        std::cerr << "Failed to open engine file for writing: " << engine_filename << std::endl;
        return -1;
    }
    engine_file.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
    std::cout << "Engine saved to " << engine_filename << std::endl;

    // 6. Clean up is now automatic thanks to std::unique_ptr.

    return 0;
}
