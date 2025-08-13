#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

namespace TrtVision {

struct Detection { cv::Rect2f box; int cls; float conf; };

// Greedy NMS (CPU): returns indices of kept detections.
// - iouTh: IoU threshold in [0,1]
// - orderByConf: if true, sort by conf desc before NMS; otherwise keep given order
std::vector<int> nms(const std::vector<Detection>& dets, float iouTh = 0.5f, bool orderByConf = true);

// Helper: IoU between two boxes
float iou(const cv::Rect2f& a, const cv::Rect2f& b);

} // namespace TrtVision
