#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

namespace TrtDecode {

struct YoloDecodeConfig {
    // If outputs are already decoded (xyxy and scores), set alreadyDecoded=true
    bool alreadyDecoded = false;
    // Layout flags for raw head: if not alreadyDecoded
    // support [N,C] or [N,C,H,W] flattened to [N,C] per image
    bool hasObjectness = true; // whether an objectness score exists at channel 4
    bool coordsAreXYWH = true; // true: [x,y,w,h], false: [x1,y1,x2,y2]
    int numClasses = 80;
};

struct YoloDet { cv::Rect2f box; int cls; float conf; };

// Minimal CPU decode for YOLO-like heads. Accepts a contiguous float buffer and dims N x C (or 2D view from [B, ...]).
// Returns detections filtered by confTh. If cfg.alreadyDecoded, expects rows as [x1,y1,x2,y2,conf,cls].
std::vector<YoloDet> decode(const float* data, int N, int C, const YoloDecodeConfig& cfg, float confTh,
                            float padX, float padY, float scale, int origW, int origH, int netW, int netH);

} // namespace TrtDecode
