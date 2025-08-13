#pragma once

#include "NvInfer.h"
#include <string>
#include <vector>

// Default entropy calibrator (EntropyCalibrator2)
class Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8Calibrator(int batchSize, int inputW, int inputH, const std::string& calibDataDirPath, 
                   const std::string& calibTableName, const char* inputBlobName, bool readCache = false);

    virtual ~Int8Calibrator();

    int getBatchSize() const noexcept override;
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
    const void* readCalibrationCache(size_t& length) noexcept override;
    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

private:
    int batchSize;
    int inputW;
    int inputH;
    int imgCount;
    std::string calibDataDirPath;
    std::string calibTableName;
    const char* inputBlobName;
    bool readCache;
    
    std::vector<std::string> imgPaths;
    size_t inputCount;
    std::vector<float> hostInput;
    void* deviceInput{nullptr};
    std::vector<char> calibCache;

    // Preprocess options (controlled via env):
    // IMAGENET_CENTER_CROP=1 -> resize short side to 256 then center crop to WxH
    // IMAGENET_NORM=1 -> apply mean/std normalization after scaling to [0,1]
    bool optCenterCrop{false};
    bool optImagenetNorm{false};
};

// Optional MinMax calibrator (enable via CALIB_ALGO=minmax)
class Int8MinMaxCalibrator : public nvinfer1::IInt8MinMaxCalibrator {
public:
    Int8MinMaxCalibrator(int batchSize, int inputW, int inputH, const std::string& calibDataDirPath,
                         const std::string& calibTableName, const char* inputBlobName, bool readCache = false);
    virtual ~Int8MinMaxCalibrator();

    int getBatchSize() const noexcept override;
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
    const void* readCalibrationCache(size_t& length) noexcept override;
    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

private:
    int batchSize;
    int inputW;
    int inputH;
    int imgCount;
    std::string calibDataDirPath;
    std::string calibTableName;
    const char* inputBlobName;
    bool readCache;

    std::vector<std::string> imgPaths;
    size_t inputCount;
    std::vector<float> hostInput;
    void* deviceInput{nullptr};
    std::vector<char> calibCache;

    bool optCenterCrop{false};
    bool optImagenetNorm{false};
};
