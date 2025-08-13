#include <trt_utils/trt_vision.h>
#include <algorithm>
#include <numeric>

namespace TrtVision {

float iou(const cv::Rect2f& a, const cv::Rect2f& b){
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return uni>0? inter/uni: 0.f;
}

std::vector<int> nms(const std::vector<Detection>& dets, float iouTh, bool orderByConf){
    std::vector<int> order(dets.size());
    std::iota(order.begin(), order.end(), 0);
    if(orderByConf){
        std::sort(order.begin(), order.end(), [&](int a,int b){ return dets[a].conf > dets[b].conf; });
    }
    std::vector<int> keep; keep.reserve(dets.size());
    std::vector<char> removed(dets.size(), 0);
    for(size_t i=0;i<order.size();++i){
        int idx = order[i]; if(removed[idx]) continue; keep.push_back(idx);
        for(size_t j=i+1;j<order.size();++j){
            int idx2 = order[j]; if(removed[idx2]) continue;
            if(iou(dets[idx].box, dets[idx2].box) > iouTh) removed[idx2] = 1;
        }
    }
    return keep;
}

} // namespace TrtVision
