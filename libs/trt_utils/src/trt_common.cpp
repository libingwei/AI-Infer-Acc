#include <trt_utils/trt_common.h>

#include <fstream>
#include <sys/stat.h>
#include <glob.h>
#include <algorithm>
#include <filesystem>

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

static std::vector<std::string> globOnce(const std::string& pattern){
    glob_t g; std::vector<std::string> out;
    if(::glob(pattern.c_str(), 0, nullptr, &g)==0){
        for(size_t i=0;i<g.gl_pathc;++i) out.emplace_back(g.gl_pathv[i]);
    }
    globfree(&g); return out;
}

std::vector<std::string> TrtHelpers::collectImages(const std::string& dir,
                                                   const std::vector<std::string>& exts,
                                                   bool recursive){
    std::vector<std::string> files;
    files.reserve(2048);
    if (!recursive) {
        for(const auto& ext: exts){
            std::string pat = dir + "/" + "*." + ext;
            auto v = globOnce(pat);
            files.insert(files.end(), v.begin(), v.end());
        }
    } else {
        std::error_code ec;
        for (auto it = std::filesystem::recursive_directory_iterator(dir, std::filesystem::directory_options::skip_permission_denied, ec);
             it != std::filesystem::end(it); ++it) {
            if (ec) break;
            if (!it->is_regular_file()) continue;
            const auto& p = it->path();
            std::string ext = p.extension().string();
            if (ext.size() > 0 && ext[0] == '.') ext.erase(0,1);
            for (const auto& e : exts) {
                if (ext == e) { files.emplace_back(p.string()); break; }
            }
        }
    }
    std::sort(files.begin(), files.end());
    files.erase(std::unique(files.begin(), files.end()), files.end());
    return files;
}
