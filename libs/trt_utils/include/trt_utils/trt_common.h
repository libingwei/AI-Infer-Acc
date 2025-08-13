#pragma once

#include <string>
#include <vector>
#include <iostream>

#include "NvInfer.h"
#include "cuda_runtime_api.h"

// Unified CUDA error check macro
#ifndef CHECK
#define CHECK(status) \
    do { \
        auto _ret = (status); \
        if (_ret != 0) { \
            std::cerr << "Cuda failure: " << cudaGetErrorString(_ret) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            abort(); \
        } \
    } while (0)
#endif

// 1) Logger with configurable severity
class TrtLogger : public nvinfer1::ILogger {
public:
    explicit TrtLogger(Severity minSeverity = Severity::kWARNING) : minSeverity_(minSeverity) {}
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= minSeverity_) std::cout << msg << std::endl;
    }
    void setMinSeverity(Severity s) { minSeverity_ = s; }
private:
    Severity minSeverity_;
};

// 2) Engine/FS helpers
class EngineIO {
public:
    static std::vector<char> readFile(const std::string& path);
    static bool writeFile(const std::string& path, const void* data, size_t size);
    static bool dirExists(const std::string& path);
};

// 3) TensorRT helpers
class TrtHelpers {
public:
    static std::string firstTensorName(const nvinfer1::ICudaEngine& engine,
                                       nvinfer1::TensorIOMode mode,
                                       const char* defaultName);
    // Collect image files under a directory root (non-recursive) with common extensions.
    // Returns sorted, de-duplicated absolute/relative paths as matched by glob.
    static std::vector<std::string> collectImages(const std::string& dir,
                                                  const std::vector<std::string>& exts = {"jpg","JPG","jpeg","JPEG","png","PNG"},
                                                  bool recursive = false);
};
