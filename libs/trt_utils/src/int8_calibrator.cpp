#include <trt_utils/int8_calibrator.h>
#include <trt_utils/trt_common.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <glob.h> // For file pattern matching
#include "cuda_runtime_api.h"

// OpenCV for image processing
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <trt_utils/trt_preprocess.h>

// glob helper moved to trt_utils::TrtHelpers::collectImages

Int8Calibrator::Int8Calibrator(int batchSize, int inputW, int inputH, const std::string& calibDataDirPath, 
                               const std::string& calibTableName, const char* inputBlobName, bool readCache)
    : batchSize(batchSize), inputW(inputW), inputH(inputH), calibDataDirPath(calibDataDirPath), 
      calibTableName(calibTableName), inputBlobName(inputBlobName), readCache(readCache) {

    // Read preprocess options from env vars
    if (const char* e = std::getenv("IMAGENET_CENTER_CROP")) {
        std::string v = e; if (v == "1" || v == "true") optCenterCrop = true;
    }
    if (const char* e = std::getenv("IMAGENET_NORM")) {
        std::string v = e; if (v == "1" || v == "true") optImagenetNorm = true;
    }

    inputCount = 3 * inputW * inputH;
    hostInput.resize(batchSize * inputCount);

    // Get all image file paths (support common extensions, case-insensitive)
    bool recursive = false; if (const char* e = std::getenv("CALIB_RECURSIVE")) { std::string v=e; if (v=="1"||v=="true") recursive=true; }
    imgPaths = TrtHelpers::collectImages(calibDataDirPath, {"jpg","JPG","jpeg","JPEG","png","PNG"}, recursive);
    if (imgPaths.empty()) {
        std::cerr << "Error: No images found in directory: " << calibDataDirPath
                  << " (supported: .jpg/.jpeg/.png, case-insensitive)" << std::endl;
        exit(1);
    }
    std::cout << "Found " << imgPaths.size() << " images for calibration." << std::endl;
    imgCount = 0;

    CHECK(cudaMalloc(&deviceInput, batchSize * inputCount * sizeof(float)));
}

Int8Calibrator::~Int8Calibrator() {
    CHECK(cudaFree(deviceInput));
}

int Int8Calibrator::getBatchSize() const noexcept {
    return batchSize;
}

bool Int8Calibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
    if (imgCount >= imgPaths.size()) {
        return false; // No more images to process
    }

    int currentBatchSize = std::min(batchSize, (int)(imgPaths.size() - imgCount));

    for (int i = 0; i < currentBatchSize; ++i) {
        cv::Mat img = cv::imread(imgPaths[imgCount + i]);
        if (img.empty()) {
            std::cerr << "Warning: Could not read image " << imgPaths[imgCount + i] << std::endl;
            continue;
        }

        // Preprocess the image via shared utilities
        PreprocOptions pp; pp.centerCrop = optCenterCrop; pp.imagenetNorm = optImagenetNorm;
        cv::Mat floatImg = preprocessImage(img, inputW, inputH, pp);

        // HWC to CHW conversion and copy to host buffer
        float* batchPtr = hostInput.data() + i * inputCount;
        hwcToChw(floatImg, batchPtr);
    }

    CHECK(cudaMemcpy(deviceInput, hostInput.data(), currentBatchSize * inputCount * sizeof(float), cudaMemcpyHostToDevice));
    
    // Find the binding for the input blob
    for (int i = 0; i < nbBindings; ++i) {
        if (strcmp(names[i], inputBlobName) == 0) {
            bindings[i] = deviceInput;
            break;
        }
    }

    imgCount += currentBatchSize;
    std::cout << "Processing calibration batch " << (imgCount / batchSize) << "/" << (imgPaths.size() / batchSize) << std::endl;
    return true;
}

const void* Int8Calibrator::readCalibrationCache(size_t& length) noexcept {
    if (!readCache) {
        return nullptr;
    }
    std::cout << "Reading calibration cache from: " << calibTableName << std::endl;
    calibCache.clear();
    std::ifstream input(calibTableName, std::ios::binary);
    input >> std::noskipws;
    if (input.good()) {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calibCache));
    }
    length = calibCache.size();
    return length ? calibCache.data() : nullptr;
}

void Int8Calibrator::writeCalibrationCache(const void* cache, size_t length) noexcept {
    std::cout << "Writing calibration cache to: " << calibTableName << " (" << length << " bytes)" << std::endl;
    std::ofstream output(calibTableName, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}
