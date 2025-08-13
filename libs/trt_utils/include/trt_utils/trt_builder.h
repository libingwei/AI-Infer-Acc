#pragma once

#include <memory>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <trt_utils/trt_common.h>

// Small facade to build a TensorRT engine from ONNX with options
struct BuildOptions {
    std::string precision;           // fp32|fp16|int8
    std::string calibDataDir;        // used when int8
    std::string calibTable = "int8_calib_table.cache";
    int maxBatch = 32;               // optimization profile max batch
    // Optional dynamic H/W optimization profiles (0 means keep from ONNX)
    int hwMinH = 0, hwMinW = 0;
    int hwOptH = 0, hwOptW = 0;
    int hwMaxH = 0, hwMaxW = 0;
};

class TrtEngineBuilder {
public:
    explicit TrtEngineBuilder(nvinfer1::ILogger& logger) : logger_(logger) {}

    // Returns serialized engine memory on success, null on failure
    std::unique_ptr<nvinfer1::IHostMemory> buildFromOnnx(const std::string& onnxPath,
                                                         const BuildOptions& opt,
                                                         int& outInputW,
                                                         int& outInputH,
                                                         std::string& outInputName,
                                                         nvinfer1::IInt8Calibrator* extCalibrator = nullptr);
private:
    nvinfer1::ILogger& logger_;
};
