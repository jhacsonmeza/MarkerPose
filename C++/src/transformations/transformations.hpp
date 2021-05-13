#pragma once

#include <opencv2/core/mat.hpp>


enum kernelShapes {
    XDIR = 0,
    YDIR = 1,
    DIAG = 2,
    ANTIDIAG = 3
};


// ------------------------------------------------------------------------------------------------
// ---------------------------------- Pixel value transformation ----------------------------------
// ------------------------------------------------------------------------------------------------
void gamma_correction(cv::InputOutputArray _im, float gamma);
void gaussianNoise(cv::InputOutputArray _im, float mean, float std);
cv::Mat getBlurKernel(int shape, int ksize);


// ------------------------------------------------------------------------------------------------
// ------------------------------- Points and image transformations -------------------------------
// ------------------------------------------------------------------------------------------------
void hflip(cv::InputOutputArray _im, cv::InputOutputArray _target);
void vflip(cv::InputOutputArray _im, cv::InputOutputArray _target);
void affine_pts(cv::InputOutputArray _im, cv::InputOutputArray _target, cv::Size sz, float tx, float ty, float theta, float scale);
void transformer_superpoint(cv::InputOutputArray _im, cv::InputOutputArray _target, float p, cv::Size sz);


// ------------------------------------------------------------------------------------------------
// -------------------------------- Mask and image transformations --------------------------------
// ------------------------------------------------------------------------------------------------
void affine_mask(cv::InputOutputArray _im, cv::InputOutputArray _mask, cv::Size sz, float tx, float ty, float theta, float scale);
void transformer_ellipseg(cv::InputOutputArray _im, cv::InputOutputArray _mask, float p, cv::Size sz);