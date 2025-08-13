#include <trt_utils/trt_decode.h>
#include <algorithm>

namespace TrtDecode {

static inline float clampf(float v, float lo, float hi){ return std::max(lo, std::min(v, hi)); }

std::vector<YoloDet> decode(const float* p, int N, int C, const YoloDecodeConfig& cfg, float confTh,
                            float padX, float padY, float scale, int origW, int origH, int netW, int netH){
    std::vector<YoloDet> out; out.reserve(N);
    if(N<=0 || C<6 || p==nullptr) return out;

    auto toOrig = [&](float x, float y){
        // reverse letterbox if coords are in net space
        float nx = (x - padX) / scale;
        float ny = (y - padY) / scale;
        nx = clampf(nx, 0.f, (float)origW-1);
        ny = clampf(ny, 0.f, (float)origH-1);
        return std::pair<float,float>(nx, ny);
    };

    for(int i=0;i<N;++i){
        if(cfg.alreadyDecoded){
            float x1 = p[i*C+0], y1 = p[i*C+1], x2 = p[i*C+2], y2 = p[i*C+3];
            float conf = p[i*C+4]; int cls = (int)std::round(p[i*C+5]);
            if(conf < confTh) continue;
            bool inNet = (x2 <= netW+2 && y2 <= netH+2);
            if(inNet){
                auto a = toOrig(x1,y1); auto b = toOrig(x2,y2);
                x1=a.first; y1=a.second; x2=b.first; y2=b.second;
            }
            out.push_back({cv::Rect2f(x1,y1,x2-x1,y2-y1), cls, conf});
        } else {
            float x = p[i*C+0], y = p[i*C+1], w = p[i*C+2], h = p[i*C+3];
            int cls=-1; float clsScore=0.f, obj=1.f; int clsStart=4;
            if(cfg.hasObjectness){ obj = p[i*C+4]; clsStart=5; }
            for(int k=clsStart;k<C;++k){ float v=p[i*C+k]; if(v>clsScore){ clsScore=v; cls=k-clsStart; } }
            float conf = obj*clsScore; if(conf<confTh) continue;
            // xywh -> xyxy in orig image
            float bx = (x - padX) / scale; float by = (y - padY) / scale;
            float bw = w / scale; float bh = h / scale;
            float x1 = clampf(bx - bw/2.f, 0.f, (float)origW-1);
            float y1 = clampf(by - bh/2.f, 0.f, (float)origH-1);
            float x2 = clampf(bx + bw/2.f, 0.f, (float)origW-1);
            float y2 = clampf(by + bh/2.f, 0.f, (float)origH-1);
            out.push_back({cv::Rect2f(x1,y1,x2-x1,y2-y1), cls, conf});
        }
    }
    return out;
}

} // namespace TrtDecode
