#pragma once

#include <memory>
#include <string>
#include <vector>
#include <numeric>

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <trt_utils/trt_common.h>

// A small runtime wrapper to deserialize engine, create context, set shapes, allocate buffers and run
class TrtRunner {
public:
    explicit TrtRunner(nvinfer1::ILogger& logger) : logger_(logger) {}

    bool loadEngineFromFile(const std::string& enginePath);

    // Set batch size, choose IO tensor names automatically if empty
    bool prepare(int batchSize, std::string inputName = std::string(), std::string outputName = std::string());

    // Overload: also set spatial H/W if provided (>0) for dynamic shape engines
    bool prepare(int batchSize, int H, int W,
                 std::string inputName = std::string(),
                 std::string outputName = std::string());

    // Run N iterations on provided host data (size must match), return ms
    float run(int iterations, const float* hostInput, float* hostOutput,
              bool useDefaultStream = false,
              bool usePinned = false,
              bool pipeline = false);

    // Query sizes
    size_t inputSize() const { return inputSize_; }
    size_t outputSize() const { return outputSize_; }
    const std::string& inputName() const { return inputName_; }
    const std::string& outputName() const { return outputName_; }

    // Dims helpers
    nvinfer1::Dims inputDims() const { return ctx_ ? ctx_->getTensorShape(inputName_.c_str()) : nvinfer1::Dims{}; }
    nvinfer1::Dims outputDims() const { return ctx_ ? ctx_->getTensorShape(outputName_.c_str()) : nvinfer1::Dims{}; }

private:
    nvinfer1::ILogger& logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx_;

    std::string inputName_;
    std::string outputName_;
    size_t inputSize_{0};
    size_t outputSize_{0};
    void* dIn_{nullptr};
    void* dOut_{nullptr};
};
