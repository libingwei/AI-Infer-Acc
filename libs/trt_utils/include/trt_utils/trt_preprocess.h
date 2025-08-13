#pragma once

#include <opencv2/opencv.hpp>

struct PreprocOptions {
    bool centerCrop = false;    // Resize shorter side to 256 then center crop to WxH
    bool imagenetNorm = false;  // Apply Imagenet mean/std after scaling to [0,1]
};

// Preprocess image to target size WxH with options; return float32 HWC in [0,1] and optionally normalized.
cv::Mat preprocessImage(const cv::Mat& src, int W, int H, const PreprocOptions& opt);

// Convert HWC float32 image to CHW contiguous layout at dst.
void hwcToChw(const cv::Mat& img, float* dst);

// Generalized preprocess with configurable mean/std and explicit RGB toggle.
// - centerCrop: if true, resize short side to 256 then center-crop WxH, else direct resize to WxH.
// - toRGB: convert BGR->RGB when true.
// - scaleTo01: divide by 255 when true.
// - mean/std: when provided (size 3), apply per-channel (img - mean) / std after optional scaling.
cv::Mat preprocessImageWithMeanStd(const cv::Mat& src, int W, int H,
                                   bool centerCrop,
                                   bool toRGB,
                                   bool scaleTo01,
                                   const cv::Scalar* mean, // nullptr to skip
                                   const cv::Scalar* stdv  // nullptr to skip
);
