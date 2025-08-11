#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <dirent.h>
#include <unordered_map>
#include <sstream>
#include <cstdlib>
#include <sys/stat.h>

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <trt_utils/trt_common.h>

#include <opencv2/opencv.hpp>
#include <trt_utils/trt_preprocess.h>

static void listImagesRec(const std::string& dir, std::vector<std::string>& files) {
    DIR* dp = opendir(dir.c_str());
    if (!dp) return;
    dirent* de;
    while ((de = readdir(dp)) != nullptr) {
        std::string name = de->d_name;
        if (name == "." || name == "..") continue;
        std::string path = dir + "/" + name;
        // Determine if directory
        bool is_dir = false;
#ifdef DT_DIR
        if (de->d_type == DT_DIR) {
            is_dir = true;
        } else if (de->d_type == DT_UNKNOWN) {
            struct stat st{};
            if (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) is_dir = true;
        }
#else
        struct stat st{};
        if (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) is_dir = true;
#endif
        if (is_dir) {
            listImagesRec(path, files);
            continue;
        }
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        // accept common image extensions: .jpg, .jpeg, .png
        bool is_jpg = (lower.size() >= 4 && lower.rfind(".jpg") == lower.size()-4) ||
                      (lower.size() >= 5 && lower.rfind(".jpeg") == lower.size()-5);
        bool is_png = (lower.size() >= 4 && lower.rfind(".png") == lower.size()-4);
        if (is_jpg || is_png) {
            files.push_back(path);
        }
    }
    closedir(dp);
}

static std::vector<std::string> listImages(const std::string& dir) {
    std::vector<std::string> files;
    listImagesRec(dir, files);
    std::sort(files.begin(), files.end());
    return files;
}

static int argmax(const std::vector<float>& v) {
    return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

static std::vector<int> topk(const std::vector<float>& v, int k) {
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin()+k, idx.end(), [&](int a, int b){return v[a] > v[b];});
    idx.resize(k);
    return idx;
}

static float cosine_sim(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    size_t n = a.size();
    for (size_t i = 0; i < n; ++i) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    if (na == 0 || nb == 0) return 0.f;
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 13) {
        std::cerr << "Usage: " << argv[0] << " <baseline_engine_fp32> <test_engine_fp16_or_int8> <images_dir> [max_images] [--labels <labels_csv>] [--class-names <tsv>] [--inspect <N>] [--center-crop] [--imagenet-norm]" << std::endl;
        return -1;
    }
    std::string base_engine_path = argv[1];
    std::string test_engine_path = argv[2];
    std::string images_dir = argv[3];
    int max_images = 200;
    std::string labels_csv;
    std::string class_names_tsv;
    int inspect_n = 0;
    // Parse optional args
    PreprocOptions pp;
    for (int i = 4; i < argc; ++i) {
        std::string tok = argv[i];
        if (tok == "--labels" && i+1 < argc) { labels_csv = argv[++i]; continue; }
    if (tok == "--center-crop") { pp.centerCrop = true; continue; }
        if (tok == "--imagenet-norm") { pp.imagenetNorm = true; continue; }
    if (tok == "--class-names" && i+1 < argc) { class_names_tsv = argv[++i]; continue; }
    if (tok == "--inspect" && i+1 < argc) { inspect_n = std::stoi(argv[++i]); continue; }
        // first free integer becomes max_images
        if (tok.size() && std::all_of(tok.begin(), tok.end(), ::isdigit)) {
            max_images = std::stoi(tok);
        }
    }
    if (labels_csv.empty()) {
        const char* env = std::getenv("LABELS_CSV");
        if (env && *env) labels_csv = env;
    }

    TrtLogger gLogger;

    auto base_data = EngineIO::readFile(base_engine_path);
    auto test_data = EngineIO::readFile(test_engine_path);
    if (base_data.empty() || test_data.empty()) { std::cerr << "Failed to read engine files" << std::endl; return -1; }

    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(gLogger));
    std::unique_ptr<nvinfer1::ICudaEngine> base_engine(runtime->deserializeCudaEngine(base_data.data(), base_data.size()));
    std::unique_ptr<nvinfer1::ICudaEngine> test_engine(runtime->deserializeCudaEngine(test_data.data(), test_data.size()));
    std::unique_ptr<nvinfer1::IExecutionContext> base_ctx(base_engine->createExecutionContext());
    std::unique_ptr<nvinfer1::IExecutionContext> test_ctx(test_engine->createExecutionContext());

    std::string base_in_name = TrtHelpers::firstTensorName(*base_engine, nvinfer1::TensorIOMode::kINPUT, "input");
    std::string base_out_name = TrtHelpers::firstTensorName(*base_engine, nvinfer1::TensorIOMode::kOUTPUT, "output");
    std::string test_in_name = TrtHelpers::firstTensorName(*test_engine, nvinfer1::TensorIOMode::kINPUT, base_in_name.c_str());
    std::string test_out_name = TrtHelpers::firstTensorName(*test_engine, nvinfer1::TensorIOMode::kOUTPUT, base_out_name.c_str());

    auto in_dims = base_engine->getTensorShape(base_in_name.c_str());
    int H = (in_dims.nbDims >= 3) ? in_dims.d[in_dims.nbDims-2] : 224;
    int W = (in_dims.nbDims >= 3) ? in_dims.d[in_dims.nbDims-1] : 224;
    in_dims.d[0] = 1; // batch 1
    base_ctx->setInputShape(base_in_name.c_str(), in_dims);
    test_ctx->setInputShape(test_in_name.c_str(), in_dims);

    // Sizes
    auto base_out_dims = base_ctx->getTensorShape(base_out_name.c_str());
    size_t base_out_size = std::accumulate(base_out_dims.d, base_out_dims.d + base_out_dims.nbDims, 1LL, std::multiplies<int64_t>());
    auto test_out_dims = test_ctx->getTensorShape(test_out_name.c_str());
    size_t test_out_size = std::accumulate(test_out_dims.d, test_out_dims.d + test_out_dims.nbDims, 1LL, std::multiplies<int64_t>());
    if (base_out_size != test_out_size) {
        std::cerr << "Output sizes differ (" << base_out_size << " vs " << test_out_size << ")" << std::endl;
        return -1;
    }

    size_t in_chw = static_cast<size_t>(3) * H * W;
    void *base_in_dev=nullptr, *base_out_dev=nullptr, *test_in_dev=nullptr, *test_out_dev=nullptr;
    CHECK(cudaMalloc(&base_in_dev, in_chw * sizeof(float)));
    CHECK(cudaMalloc(&test_in_dev, in_chw * sizeof(float)));
    CHECK(cudaMalloc(&base_out_dev, base_out_size * sizeof(float)));
    CHECK(cudaMalloc(&test_out_dev, test_out_size * sizeof(float)));

    base_ctx->setTensorAddress(base_in_name.c_str(), base_in_dev);
    base_ctx->setTensorAddress(base_out_name.c_str(), base_out_dev);
    test_ctx->setTensorAddress(test_in_name.c_str(), test_in_dev);
    test_ctx->setTensorAddress(test_out_name.c_str(), test_out_dev);

    cudaStream_t stream; CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // List images
    auto images = listImages(images_dir);
    if (images.empty()) { std::cerr << "No images in dir: " << images_dir << std::endl; return -1; }
    if ((int)images.size() > max_images) images.resize(max_images);
    std::cout << "Evaluating on " << images.size() << " images (H=" << H << ", W=" << W << ")\n";

    // Load optional labels CSV: filename,labelIndex(0-based)
    std::unordered_map<std::string,int> labels;
    if (!labels_csv.empty()) {
        std::ifstream lf(labels_csv);
        if (!lf) {
            std::cerr << "Failed to open labels CSV: " << labels_csv << std::endl;
            return -1;
        }
        std::string line;
        while (std::getline(lf, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string fname, labstr;
            if (!std::getline(ss, fname, ',')) continue;
            if (!std::getline(ss, labstr)) continue;
            int lab = std::stoi(labstr);
            labels[fname] = lab;
        }
        std::cout << "Loaded labels from " << labels_csv << ", entries: " << labels.size() << "\n";
    }

    auto basename = [](const std::string& p){
        size_t pos = p.find_last_of("/\\");
        return (pos==std::string::npos) ? p : p.substr(pos+1);
    };

    // Optional: load class names mapping (index -> readable name)
    std::vector<std::string> class_names;
    if (!class_names_tsv.empty()) {
        std::ifstream cf(class_names_tsv);
        if (!cf) {
            std::cerr << "Failed to open class names TSV: " << class_names_tsv << std::endl;
            return -1;
        }
        std::string line;
        while (std::getline(cf, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string idxs, wnid, name;
            if (!std::getline(ss, idxs, '\t')) continue;
            if (!std::getline(ss, wnid, '\t')) continue;
            if (!std::getline(ss, name)) name = wnid;
            int idx = std::stoi(idxs);
            if ((int)class_names.size() <= idx) class_names.resize(idx+1);
            class_names[idx] = name;
        }
        std::cout << "Loaded class names: " << class_names.size() << "\n";
    }

    // Metrics accumulators (consistency)
    size_t n = 0, agree_top1 = 0, agree_top5 = 0;
    double sum_cos = 0.0, sum_l2 = 0.0;

    // Accuracy accumulators (labeled)
    size_t labeled_n = 0;
    size_t base_top1 = 0, base_top5 = 0;
    size_t test_top1 = 0, test_top5 = 0;

    std::vector<float> host_in(in_chw);
    std::vector<float> base_out(base_out_size), test_out(test_out_size);

    for (const auto& path : images) {
        cv::Mat img = cv::imread(path);
        if (img.empty()) continue;
        cv::Mat proc = preprocessImage(img, W, H, pp);
        hwcToChw(proc, host_in.data());

        // Copy to both inputs (same tensor layout)
        CHECK(cudaMemcpyAsync(base_in_dev, host_in.data(), in_chw*sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(test_in_dev, host_in.data(), in_chw*sizeof(float), cudaMemcpyHostToDevice, stream));

        // Run both engines sequentially on the same stream
        base_ctx->enqueueV3(stream);
        CHECK(cudaMemcpyAsync(base_out.data(), base_out_dev, base_out_size*sizeof(float), cudaMemcpyDeviceToHost, stream));

        test_ctx->enqueueV3(stream);
        CHECK(cudaMemcpyAsync(test_out.data(), test_out_dev, test_out_size*sizeof(float), cudaMemcpyDeviceToHost, stream));

        CHECK(cudaStreamSynchronize(stream));

    int b1 = argmax(base_out);
    int t1 = argmax(test_out);
        if (b1 == t1) ++agree_top1;

        auto b5 = topk(base_out, 5);
        auto t5 = topk(test_out, 5);
        bool hasOverlap = false;
        for (int bi : b5) {
            if (std::find(t5.begin(), t5.end(), bi) != t5.end()) { hasOverlap = true; break; }
        }
        if (hasOverlap) ++agree_top5;

        float cosv = cosine_sim(base_out, test_out);
        sum_cos += cosv;
        // L2 distance
        double l2=0.0; for (size_t i=0;i<base_out.size();++i){ double d = (double)base_out[i]-test_out[i]; l2 += d*d; }
        sum_l2 += std::sqrt(l2);

        // Optional inspection: print first N predictions with names
        if (inspect_n > 0 && (int)n < inspect_n) {
            std::string bn = basename(path);
            auto toName = [&](int idx){
                if (!class_names.empty() && idx >= 0 && idx < (int)class_names.size() && !class_names[idx].empty()) return class_names[idx];
                return std::to_string(idx);
            };
            auto topk_names = [&](const std::vector<float>& v, int k){
                auto idxs = topk(v, k);
                std::stringstream s;
                for (size_t i=0;i<idxs.size();++i){ if(i) s<<", "; s<<idxs[i]<<":"<<toName(idxs[i]); }
                return s.str();
            };
            std::cout << "[Inspect] " << bn << " | FP32: " << b1 << ":" << toName(b1)
                      << " | Test: " << t1 << ":" << toName(t1)
                      << " | Top5(FP32): " << topk_names(base_out,5)
                      << " | Top5(Test): " << topk_names(test_out,5) << "\n";
        }

        // Labeled accuracy
        if (!labels.empty()) {
            auto it = labels.find(basename(path));
            if (it != labels.end()) {
                int gt = it->second; // 0-based class index
                if (b1 == gt) ++base_top1;
                if (t1 == gt) ++test_top1;
                auto contains = [](const std::vector<int>& vv, int x){ return std::find(vv.begin(), vv.end(), x) != vv.end(); };
                if (contains(b5, gt)) ++base_top5;
                if (contains(t5, gt)) ++test_top5;
                ++labeled_n;
            }
        }

        ++n;
    }

    std::cout << "\n=== Accuracy Consistency Report ===\n";
    std::cout << "Samples: " << n << "\n";
    if (n>0) {
        std::cout << "Top-1 Agreement: " << (100.0*agree_top1/n) << "%\n";
        std::cout << "Top-5 Agreement: " << (100.0*agree_top5/n) << "%\n";
        std::cout << "Avg Cosine Similarity: " << (sum_cos/n) << "\n";
        std::cout << "Avg L2 Distance: " << (sum_l2/n) << "\n";
    }

    if (!labels.empty()) {
        std::cout << "\n=== Labeled Accuracy Report ===\n";
        std::cout << "Labeled Samples: " << labeled_n << "\n";
        if (labeled_n>0) {
            double base_top1_acc = 100.0*base_top1/labeled_n;
            double base_top5_acc = 100.0*base_top5/labeled_n;
            double test_top1_acc = 100.0*test_top1/labeled_n;
            double test_top5_acc = 100.0*test_top5/labeled_n;
            std::cout << "FP32 Top-1: " << base_top1_acc << "%, Top-5: " << base_top5_acc << "%\n";
            std::cout << "Test Top-1: " << test_top1_acc << "%, Top-5: " << test_top5_acc << "%\n";
            std::cout << "Delta Top-1: " << (test_top1_acc - base_top1_acc) << " pp, Delta Top-5: " << (test_top5_acc - base_top5_acc) << " pp\n";
        }
    }

    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(base_in_dev));
    CHECK(cudaFree(test_in_dev));
    CHECK(cudaFree(base_out_dev));
    CHECK(cudaFree(test_out_dev));
    return 0;
}
