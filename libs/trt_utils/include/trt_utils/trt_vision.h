#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

namespace TrtVision {

struct Detection { cv::Rect2f box; int cls; float conf; };

// Greedy NMS (CPU): returns indices of kept detections.
// - iouTh: IoU threshold in [0,1]
// - orderByConf: if true, sort by conf desc before NMS; otherwise keep given order
// Class-agnostic NMS; if classAgnostic=false, NMS is done per class.
// topK: if >0, keep at most topK results after NMS.
std::vector<int> nms(const std::vector<Detection>& dets, float iouTh = 0.5f, bool orderByConf = true,
					 bool classAgnostic = true, int topK = -1);

// Batched NMS: for each image's detections, apply NMS independently, returning indices per image.
std::vector<std::vector<int>> nmsBatched(const std::vector<std::vector<Detection>>& batchDets,
										 float iouTh = 0.5f, bool orderByConf = true,
										 bool classAgnostic = true, int topK = -1);

// Helper: IoU between two boxes
float iou(const cv::Rect2f& a, const cv::Rect2f& b);

} // namespace TrtVision
