#include <trt_utils/trt_builder.h>
#include <trt_utils/int8_calibrator.h>

#include <cstdlib>
#include <iostream>

std::unique_ptr<nvinfer1::IHostMemory> TrtEngineBuilder::buildFromOnnx(const std::string& onnxPath,
                                                                        const BuildOptions& opt,
                                                                        int& outInputW,
                                                                        int& outInputH,
                                                                        std::string& outInputName,
                                                                        nvinfer1::IInt8Calibrator* extCalibrator) {
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger_));
    if (!builder) { std::cerr << "Failed to create builder" << std::endl; return nullptr; }
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger_));

    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        std::cerr << "Failed to parse ONNX: " << onnxPath << std::endl;
        return nullptr;
    }

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL<<30); // 1GB

    std::unique_ptr<Int8Calibrator> calibrator;

    if (opt.precision == "fp16") {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "Build FP16" << std::endl;
    } else if (opt.precision == "int8") {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        std::cout << "Build INT8" << std::endl;
        // Determine calib dir
        std::string calibDir = opt.calibDataDir;
        if (calibDir.empty()) {
            if (const char* e = std::getenv("CALIB_DATA_DIR")) calibDir = e; else calibDir = "calibration_data";
        }
        const char* inName = network->getInput(0)->getName();
        auto dims = network->getInput(0)->getDimensions();
        int nb = dims.nbDims;
        outInputW = (nb>=2) ? dims.d[nb-1] : 224;
        outInputH = (nb>=2) ? dims.d[nb-2] : 224;
        if (outInputW <= 0 || outInputH <= 0) { outInputW = 224; outInputH = 224; }
        outInputName = inName;
        if (extCalibrator) {
            config->setInt8Calibrator(extCalibrator);
        } else {
            if (!EngineIO::dirExists(calibDir)) {
                std::cerr << "Calibration directory missing: " << calibDir << std::endl;
                return nullptr;
            }
            calibrator = std::make_unique<Int8Calibrator>(8, outInputW, outInputH, calibDir, opt.calibTable, inName);
            config->setInt8Calibrator(calibrator.get());
        }
    } else {
        std::cout << "Build FP32" << std::endl;
    }

    // Opt profile
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    outInputName = network->getInput(0)->getName();
    auto input_dims = network->getInput(0)->getDimensions();
    input_dims.d[0] = 1; profile->setDimensions(outInputName.c_str(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    input_dims.d[0] = 1; profile->setDimensions(outInputName.c_str(), nvinfer1::OptProfileSelector::kOPT, input_dims);
    input_dims.d[0] = opt.maxBatch; profile->setDimensions(outInputName.c_str(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);

    auto ser = builder->buildSerializedNetwork(*network, *config);
    if (!ser) { std::cerr << "Failed to build engine" << std::endl; return nullptr; }
    return std::unique_ptr<nvinfer1::IHostMemory>(ser);
}
