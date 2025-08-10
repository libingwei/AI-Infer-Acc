// src/02_onnx_to_trt.cpp
//
// Purpose: Converts a given ONNX model file into a serialized TensorRT engine.
//
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <cstdlib>

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <trt_utils/int8_calibrator.h>
#include <trt_utils/trt_builder.h>
#include <trt_utils/trt_common.h>

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_onnx_path> <output_base_name> <precision> [calib_data_dir]" << std::endl;
        std::cerr << "  <precision> can be: fp32, fp16, int8" << std::endl;
        return -1;
    }
    const char* onnx_filename = argv[1];
    const char* output_basename = argv[2];
    std::string precision = argv[3];

    TrtLogger gLogger;

    BuildOptions opt; opt.precision = precision; if (argc>=5) opt.calibDataDir = argv[4];
    int W=0,H=0; std::string inName;
    TrtEngineBuilder builder(gLogger);
    auto ser = builder.buildFromOnnx(onnx_filename, opt, W, H, inName);
    if (!ser) { std::cerr << "Failed to build engine." << std::endl; return -1; }

    std::string engine_filename = std::string(output_basename);
    if (precision == "fp16" || precision == "int8") engine_filename += "_" + precision;
    engine_filename += ".trt";

    if (!EngineIO::writeFile(engine_filename, ser->data(), ser->size())) {
        std::cerr << "Failed to write engine file: " << engine_filename << std::endl; return -1;
    }
    std::cout << "Saved: " << engine_filename << std::endl;
    return 0;
}
