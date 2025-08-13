#include <trt_utils/trt_vision.h>
#include <algorithm>
#include <numeric>

namespace TrtVision {

float iou(const cv::Rect2f& a, const cv::Rect2f& b){
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return uni>0? inter/uni: 0.f;
}

static std::vector<int> _nms_single(const std::vector<Detection>& dets, float iouTh, bool orderByConf, bool classAgnostic, int topK){
    std::vector<int> order(dets.size());
    std::iota(order.begin(), order.end(), 0);
    if(orderByConf){
        std::sort(order.begin(), order.end(), [&](int a,int b){ return dets[a].conf > dets[b].conf; });
    }
    std::vector<int> keep; keep.reserve(dets.size());
    std::vector<char> removed(dets.size(), 0);
    for(size_t i=0;i<order.size();++i){
        int idx = order[i]; if(removed[idx]) continue; keep.push_back(idx);
        if(topK>0 && (int)keep.size()>=topK) break;
        for(size_t j=i+1;j<order.size();++j){
            int idx2 = order[j]; if(removed[idx2]) continue;
            if(!classAgnostic && dets[idx].cls != dets[idx2].cls) continue;
            if(iou(dets[idx].box, dets[idx2].box) > iouTh) removed[idx2] = 1;
        }
    }
    return keep;
}

std::vector<int> nms(const std::vector<Detection>& dets, float iouTh, bool orderByConf, bool classAgnostic, int topK){
    return _nms_single(dets, iouTh, orderByConf, classAgnostic, topK);
}

std::vector<std::vector<int>> nmsBatched(const std::vector<std::vector<Detection>>& batchDets,
                                         float iouTh, bool orderByConf, bool classAgnostic, int topK){
    std::vector<std::vector<int>> res; res.reserve(batchDets.size());
    for(const auto& dets: batchDets){ res.push_back(_nms_single(dets, iouTh, orderByConf, classAgnostic, topK)); }
    return res;
}

} // namespace TrtVision
