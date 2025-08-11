#include <trt_utils/trt_runtime.h>

#include <chrono>

bool TrtRunner::loadEngineFromFile(const std::string& enginePath) {
    auto data = EngineIO::readFile(enginePath);
    if (data.empty()) return false;
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) return false;
    engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
    if (!engine_) return false;
    ctx_.reset(engine_->createExecutionContext());
    return ctx_ != nullptr;
}

bool TrtRunner::prepare(int batchSize, std::string inputName, std::string outputName) {
    if (!engine_ || !ctx_) return false;
    if (inputName.empty()) inputName_ = TrtHelpers::firstTensorName(*engine_, nvinfer1::TensorIOMode::kINPUT, "input");
    else inputName_ = std::move(inputName);
    if (outputName.empty()) outputName_ = TrtHelpers::firstTensorName(*engine_, nvinfer1::TensorIOMode::kOUTPUT, "output");
    else outputName_ = std::move(outputName);

    auto inDims = engine_->getTensorShape(inputName_.c_str());
    inDims.d[0] = batchSize;
    if (!ctx_->setInputShape(inputName_.c_str(), inDims)) return false;

    auto fin = ctx_->getTensorShape(inputName_.c_str());
    inputSize_ = 1; for (int i=0;i<fin.nbDims;++i) inputSize_ *= fin.d[i];
    auto fout = ctx_->getTensorShape(outputName_.c_str());
    outputSize_ = 1; for (int i=0;i<fout.nbDims;++i) outputSize_ *= fout.d[i];

    CHECK(cudaMalloc(&dIn_, inputSize_ * sizeof(float)));
    CHECK(cudaMalloc(&dOut_, outputSize_ * sizeof(float)));

    ctx_->setTensorAddress(inputName_.c_str(), dIn_);
    ctx_->setTensorAddress(outputName_.c_str(), dOut_);

    return true;
}

bool TrtRunner::prepare(int batchSize, int H, int W,
                        std::string inputName, std::string outputName) {
    if (!engine_ || !ctx_) return false;
    if (inputName.empty()) inputName_ = TrtHelpers::firstTensorName(*engine_, nvinfer1::TensorIOMode::kINPUT, "input");
    else inputName_ = std::move(inputName);
    if (outputName.empty()) outputName_ = TrtHelpers::firstTensorName(*engine_, nvinfer1::TensorIOMode::kOUTPUT, "output");
    else outputName_ = std::move(outputName);

    auto inDims = engine_->getTensorShape(inputName_.c_str());
    inDims.d[0] = batchSize;
    if (inDims.nbDims >= 4) {
        if (H > 0) inDims.d[2] = H;
        if (W > 0) inDims.d[3] = W;
    }
    if (!ctx_->setInputShape(inputName_.c_str(), inDims)) return false;

    auto fin = ctx_->getTensorShape(inputName_.c_str());
    inputSize_ = 1; for (int i=0;i<fin.nbDims;++i) inputSize_ *= std::max<int64_t>(1, fin.d[i]);
    auto fout = ctx_->getTensorShape(outputName_.c_str());
    outputSize_ = 1; for (int i=0;i<fout.nbDims;++i) outputSize_ *= std::max<int64_t>(1, fout.d[i]);

    CHECK(cudaMalloc(&dIn_, inputSize_ * sizeof(float)));
    CHECK(cudaMalloc(&dOut_, outputSize_ * sizeof(float)));

    ctx_->setTensorAddress(inputName_.c_str(), dIn_);
    ctx_->setTensorAddress(outputName_.c_str(), dOut_);
    return true;
}

float TrtRunner::run(int iterations, const float* hostInput, float* hostOutput,
                     bool useDefaultStream, bool usePinned, bool pipeline) {
    float* hIn = const_cast<float*>(hostInput);
    float* hOut = hostOutput;

    if (!pipeline) {
        // Single-stream path (default or non-default stream)
        // Use nullptr for default stream; create a non-default stream only when requested
        cudaStream_t stream = nullptr;
        if (!useDefaultStream) CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        // Preload input and warm-up
        CHECK(cudaMemcpyAsync(dIn_, hIn, inputSize_*sizeof(float), cudaMemcpyHostToDevice, stream));
        for (int i = 0; i < 10; ++i) ctx_->enqueueV3(stream);
        CHECK(cudaStreamSynchronize(stream));

        // Timed runs
        cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
        CHECK(cudaEventRecord(s, stream));
        for (int i=0;i<iterations;++i) ctx_->enqueueV3(stream);
        CHECK(cudaEventRecord(e, stream));
        CHECK(cudaEventSynchronize(e));

        // Copy back one output snapshot
        CHECK(cudaMemcpyAsync(hOut, dOut_, outputSize_*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));

        float ms=0.f; CHECK(cudaEventElapsedTime(&ms, s, e));
        CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));
        if (!useDefaultStream) CHECK(cudaStreamDestroy(stream));
        return ms;
    }

    // Pipeline path: two non-default streams for copy/compute and double buffers
    cudaStream_t copyStream = nullptr, computeStream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));

    void* dIn[2] {nullptr, nullptr};
    void* dOut[2] {nullptr, nullptr};
    CHECK(cudaMalloc(&dIn[0], inputSize_*sizeof(float)));
    CHECK(cudaMalloc(&dIn[1], inputSize_*sizeof(float)));
    CHECK(cudaMalloc(&dOut[0], outputSize_*sizeof(float)));
    CHECK(cudaMalloc(&dOut[1], outputSize_*sizeof(float)));

    auto setBuf = [&](int idx){
        ctx_->setTensorAddress(inputName_.c_str(), dIn[idx]);
        ctx_->setTensorAddress(outputName_.c_str(), dOut[idx]);
    };

    // Prime first batch
    CHECK(cudaMemcpyAsync(dIn[0], hIn, inputSize_*sizeof(float), cudaMemcpyHostToDevice, copyStream));
    CHECK(cudaStreamSynchronize(copyStream));

    // Warm-up one iteration
    setBuf(0);
    ctx_->enqueueV3(computeStream);
    CHECK(cudaStreamSynchronize(computeStream));

    // Timed loop
    cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
    CHECK(cudaEventRecord(s, computeStream));
    int cur = 0;
    for (int i = 0; i < iterations; ++i) {
        int nxt = 1 - cur;
        // Launch compute on current buffer
        setBuf(cur);
        ctx_->enqueueV3(computeStream);
        // H2D next batch in parallel
        CHECK(cudaMemcpyAsync(dIn[nxt], hIn, inputSize_*sizeof(float), cudaMemcpyHostToDevice, copyStream));
        // Wait for compute to finish before swapping
        CHECK(cudaStreamSynchronize(computeStream));
        cur = nxt;
    }
    CHECK(cudaEventRecord(e, computeStream));
    CHECK(cudaEventSynchronize(e));

    // Copy last output back
    int last = 1 - cur; // last executed buffer index
    CHECK(cudaMemcpyAsync(hOut, dOut[last], outputSize_*sizeof(float), cudaMemcpyDeviceToHost, computeStream));
    CHECK(cudaStreamSynchronize(computeStream));

    // Restore original bindings
    ctx_->setTensorAddress(inputName_.c_str(), dIn_);
    ctx_->setTensorAddress(outputName_.c_str(), dOut_);

    float ms=0.f; CHECK(cudaEventElapsedTime(&ms, s, e));
    CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));

    CHECK(cudaFree(dIn[0])); CHECK(cudaFree(dIn[1]));
    CHECK(cudaFree(dOut[0])); CHECK(cudaFree(dOut[1]));
    CHECK(cudaStreamDestroy(copyStream));
    CHECK(cudaStreamDestroy(computeStream));
    return ms;
}
