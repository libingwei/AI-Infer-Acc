#include <trt_utils/trt_preprocess.h>

#include <algorithm>

cv::Mat preprocessImage(const cv::Mat& src, int W, int H, const PreprocOptions& opt) {
    cv::Mat img = src;
    if (opt.centerCrop) {
        int shortSide = 256;
        int ih = img.rows, iw = img.cols;
        float scale = (iw < ih) ? (shortSide / static_cast<float>(iw)) : (shortSide / static_cast<float>(ih));
        int newW = static_cast<int>(std::round(iw * scale));
        int newH = static_cast<int>(std::round(ih * scale));
        cv::resize(img, img, cv::Size(newW, newH));
        int x = (newW - W) / 2;
        int y = (newH - H) / 2;
        x = std::max(0, x); y = std::max(0, y);
        x = std::min(x, std::max(0, newW - W));
        y = std::min(y, std::max(0, newH - H));
        cv::Rect roi(x, y, std::min(W, newW), std::min(H, newH));
        img = img(roi).clone();
        if (img.cols != W || img.rows != H) cv::resize(img, img, cv::Size(W, H));
    } else {
        cv::resize(img, img, cv::Size(W, H));
    }

    img.convertTo(img, CV_32FC3, 1.0/255.0);

    if (opt.imagenetNorm) {
        const cv::Scalar mean(0.485, 0.456, 0.406);
        const cv::Scalar stdv(0.229, 0.224, 0.225);
        cv::Mat meanMat(img.size(), img.type(), mean);
        cv::Mat stdMat(img.size(), img.type(), stdv);
        cv::subtract(img, meanMat, img);
        cv::divide(img, stdMat, img);
    }
    return img;
}

void hwcToChw(const cv::Mat& img, float* dst) {
    std::vector<cv::Mat> ch(3);
    cv::split(img, ch);
    int H = img.rows, W = img.cols;
    size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);
    memcpy(dst, ch[0].data, plane * sizeof(float));
    memcpy(dst + plane, ch[1].data, plane * sizeof(float));
    memcpy(dst + 2 * plane, ch[2].data, plane * sizeof(float));
}
