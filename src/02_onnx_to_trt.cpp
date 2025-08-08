// src/02_onnx_to_trt.cpp
//
// Purpose: Converts a given ONNX model file into a serialized TensorRT engine.
// This is a crucial step for deployment, as the engine is highly optimized
// for the specific GPU hardware it's built on.

#include <iostream>
#include <fstream>
#include <memory>
#include <string>

#include "NvInfer.h"
#include "NvOnnxParser.h"

// A simple logger class required by the TensorRT API.
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_onnx_path> <output_base_name> <precision>" << std::endl;
        std::cerr << "  <precision> can be: fp32, fp16, int8" << std::endl;
        return -1;
    }
    const char* onnx_filename = argv[1];
    const char* output_basename = argv[2];
    std::string precision = argv[3];

    Logger gLogger;

    // 1. Create core TensorRT objects
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(gLogger));
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, gLogger));

    // 2. Parse the ONNX model
    if (!parser->parseFromFile(onnx_filename, static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        std::cerr << "Failed to parse ONNX file: " << onnx_filename << std::endl;
        return -1;
    }

    // 3. Configure the builder based on precision
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30); // 1GB

    if (precision == "fp16") {
        std::cout << "Building in FP16 mode." << std::endl;
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else if (precision == "int8") {
        std::cout << "INT8 mode to be implemented. Exiting." << std::endl;
        // Placeholder for future INT8 calibrator implementation
        return 0; 
    } else {
        std::cout << "Building in FP32 mode." << std::endl;
    }

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    const char* input_name = "input";
    auto input_dims = network->getInput(0)->getDimensions();

    input_dims.d[0] = 1;
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, input_dims);
    input_dims.d[0] = 1;
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, input_dims);
    input_dims.d[0] = 32;
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);

    // 4. Build the serialized engine
    std::cout << "Building TensorRT engine... (This may take a few minutes)" << std::endl;
    std::unique_ptr<nvinfer1::IHostMemory> serialized_engine(builder->buildSerializedNetwork(*network, *config));
    if (!serialized_engine) {
        std::cerr << "Failed to build engine." << std::endl;
        return -1;
    }
    std::cout << "Engine built successfully." << std::endl;

    // 5. Generate filename and save the engine
    std::string engine_filename = std::string(output_basename);
    if (precision == "fp16" || precision == "int8") {
        engine_filename += "_" + precision;
    }
    engine_filename += ".trt";

    std::ofstream engine_file(engine_filename, std::ios::binary);
    if (!engine_file) {
        std::cerr << "Failed to open engine file for writing: " << engine_filename << std::endl;
        return -1;
    }
    engine_file.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
    std::cout << "Engine saved to " << engine_filename << std::endl;

    return 0;
}
