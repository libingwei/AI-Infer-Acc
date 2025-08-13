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

    std::unique_ptr<nvinfer1::IInt8Calibrator> calibHolder;

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
            std::string algo = "entropy"; if (const char* e = std::getenv("CALIB_ALGO")) algo = e;
            for (auto& c : algo) c = static_cast<char>(::tolower(c));
            if (algo == "minmax") {
                std::cout << "Using MinMax calibrator" << std::endl;
        calibHolder.reset(new Int8MinMaxCalibrator(8, outInputW, outInputH, calibDir, opt.calibTable, inName));
        config->setInt8Calibrator(calibHolder.get());
            } else {
                std::cout << "Using Entropy calibrator" << std::endl;
        calibHolder = std::make_unique<Int8Calibrator>(8, outInputW, outInputH, calibDir, opt.calibTable, inName);
        config->setInt8Calibrator(calibHolder.get());
            }
        }

        // Optional mixed-precision protections via env flags
        bool protectFirst = false, protectLast = false, protectSoftmax = false;
        if (const char* e = std::getenv("INT8_FP16_FIRST")) { std::string v=e; if (v=="1"||v=="true") protectFirst=true; }
        if (const char* e = std::getenv("INT8_FP16_LAST")) { std::string v=e; if (v=="1"||v=="true") protectLast=true; }
        if (const char* e = std::getenv("INT8_FP32_SOFTMAX")) { std::string v=e; if (v=="1"||v=="true") protectSoftmax=true; }

        // Per-layer overrides by name (comma-separated substrings)
        auto splitCsv = [](const std::string& s) {
            std::vector<std::string> out; std::string cur; for (char c: s){ if(c==','){ if(!cur.empty()) out.push_back(cur); cur.clear(); } else cur.push_back(c);} if(!cur.empty()) out.push_back(cur); return out; };
        std::vector<std::string> forceHalfNames, forceFloatNames;
        if (const char* e = std::getenv("INT8_FORCE_FP16_LAYERS")) forceHalfNames = splitCsv(e);
        if (const char* e = std::getenv("INT8_FORCE_FP32_LAYERS")) forceFloatNames = splitCsv(e);
        bool anyForced = !forceHalfNames.empty() || !forceFloatNames.empty();
        bool verboseLayers = false; if (const char* e = std::getenv("INT8_VERBOSE_LAYERS")) { std::string v=e; if (v=="1"||v=="true") verboseLayers=true; }

        if (protectFirst || protectLast || protectSoftmax || anyForced) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16); // allow FP16 when needed
            config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
            config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
            int nbLayers = network->getNbLayers();

            if (protectFirst && nbLayers > 0) {
                auto* l0 = network->getLayer(0);
                l0->setPrecision(nvinfer1::DataType::kHALF);
                int nOuts = l0->getNbOutputs();
                for (int i=0;i<nOuts;++i) l0->setOutputType(i, nvinfer1::DataType::kHALF);
            }
            if (protectLast && nbLayers > 0) {
                auto* ln = network->getLayer(nbLayers - 1);
                ln->setPrecision(nvinfer1::DataType::kHALF);
                int nOuts = ln->getNbOutputs();
                for (int i=0;i<nOuts;++i) ln->setOutputType(i, nvinfer1::DataType::kHALF);
            }
            if (protectSoftmax) {
                for (int i = 0; i < nbLayers; ++i) {
                    auto* lyr = network->getLayer(i);
                    if (lyr->getType() == nvinfer1::LayerType::kSOFTMAX) {
                        lyr->setPrecision(nvinfer1::DataType::kFLOAT);
                        int nOuts = lyr->getNbOutputs();
                        for (int j=0;j<nOuts;++j) lyr->setOutputType(j, nvinfer1::DataType::kFLOAT);
                    }
                }
            }
            if (anyForced) {
                for (int i = 0; i < nbLayers; ++i) {
                    auto* lyr = network->getLayer(i);
                    const char* nmC = lyr->getName();
                    std::string nm = nmC ? std::string(nmC) : std::string();
                    auto matchAny = [&](const std::vector<std::string>& keys){ for (auto& k: keys){ if(!k.empty() && nm.find(k) != std::string::npos) return true; } return false; };
                    if (matchAny(forceHalfNames)) {
                        if (verboseLayers) std::cout << "[INT8] Force FP16 layer: " << nm << std::endl;
                        lyr->setPrecision(nvinfer1::DataType::kHALF);
                        int nOuts = lyr->getNbOutputs();
                        for (int j=0;j<nOuts;++j) lyr->setOutputType(j, nvinfer1::DataType::kHALF);
                    }
                    if (matchAny(forceFloatNames)) {
                        if (verboseLayers) std::cout << "[INT8] Force FP32 layer: " << nm << std::endl;
                        lyr->setPrecision(nvinfer1::DataType::kFLOAT);
                        int nOuts = lyr->getNbOutputs();
                        for (int j=0;j<nOuts;++j) lyr->setOutputType(j, nvinfer1::DataType::kFLOAT);
                    }
                }
            }
        }
    } else {
        std::cout << "Build FP32" << std::endl;
    }

    // Opt profile with optional dynamic H/W
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    outInputName = network->getInput(0)->getName();
    auto input_dims = network->getInput(0)->getDimensions();
    // Expect NCHW; protect against -1 (dynamic) dimensions
    auto setDims = [&](nvinfer1::OptProfileSelector sel, int n, int c, int h, int w){
        auto d = input_dims;
        if (n > 0) d.d[0] = n;
        if (d.nbDims >= 4) {
            if (c > 0) d.d[1] = c;
            if (h > 0) d.d[2] = h;
            if (w > 0) d.d[3] = w;
        }
        profile->setDimensions(outInputName.c_str(), sel, d);
    };
    // MIN
    int minH = opt.hwMinH > 0 ? opt.hwMinH : (input_dims.nbDims>=4 ? input_dims.d[2] : -1);
    int minW = opt.hwMinW > 0 ? opt.hwMinW : (input_dims.nbDims>=4 ? input_dims.d[3] : -1);
    setDims(nvinfer1::OptProfileSelector::kMIN, 1, (input_dims.nbDims>=4? input_dims.d[1]: -1), minH, minW);
    // OPT
    int optH = opt.hwOptH > 0 ? opt.hwOptH : minH;
    int optW = opt.hwOptW > 0 ? opt.hwOptW : minW;
    setDims(nvinfer1::OptProfileSelector::kOPT, 1, (input_dims.nbDims>=4? input_dims.d[1]: -1), optH, optW);
    // MAX
    int maxH = opt.hwMaxH > 0 ? opt.hwMaxH : optH;
    int maxW = opt.hwMaxW > 0 ? opt.hwMaxW : optW;
    setDims(nvinfer1::OptProfileSelector::kMAX, opt.maxBatch, (input_dims.nbDims>=4? input_dims.d[1]: -1), maxH, maxW);
    config->addOptimizationProfile(profile);

    auto ser = builder->buildSerializedNetwork(*network, *config);
    if (!ser) { std::cerr << "Failed to build engine" << std::endl; return nullptr; }
    return std::unique_ptr<nvinfer1::IHostMemory>(ser);
}
