#include <trt_utils/trt_common.h>

#include <fstream>
#include <sys/stat.h>

std::vector<char> EngineIO::readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return {};
    }
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> data(sz);
    f.read(data.data(), sz);
    return data;
}

bool EngineIO::writeFile(const std::string& path, const void* data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open file for write: " << path << std::endl;
        return false;
    }
    f.write(reinterpret_cast<const char*>(data), size);
    return true;
}

bool EngineIO::dirExists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) return false;
    return (info.st_mode & S_IFDIR) != 0;
}

std::string TrtHelpers::firstTensorName(const nvinfer1::ICudaEngine& engine,
                                        nvinfer1::TensorIOMode mode,
                                        const char* defaultName) {
    int nb = engine.getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
        const char* name = engine.getIOTensorName(i);
        if (engine.getTensorIOMode(name) == mode) return std::string(name);
    }
    return std::string(defaultName);
}
