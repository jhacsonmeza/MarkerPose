#include "transformations.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


// ------------------------------------------------------------------------------------------------
// -------------------------------- Mask and image transformations --------------------------------
// ------------------------------------------------------------------------------------------------
void affine_mask(cv::InputOutputArray _im, cv::InputOutputArray _mask, cv::Size sz, float tx, float ty, float theta, float scale)
{
    cv::Point2f center = (cv::Point2f(sz)-cv::Point2f(1.f,1.f))/2;
    cv::Mat last_row = (cv::Mat_<double>(1,3) << 0, 0, 1);


    // Create translation transformation
    cv::Mat Ht = (cv::Mat_<double>(3,3) << 1, 0, tx, 0, 1, ty, 0, 0, 1);

    // Create rotation transformation
    cv::Mat Hr = cv::getRotationMatrix2D(center, theta, 1);
    cv::vconcat(Hr, last_row, Hr);

    // Create scaling transformation
    cv::Mat Hs = cv::getRotationMatrix2D(center, 0, scale);
    cv::vconcat(Hs, last_row, Hs);


    // Merge transformations
    cv::Mat M = Hs*Hr*Ht;

    // Apply final transformation to image and mask
    cv::warpAffine(_im, _im, M.rowRange(0,2), sz, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    cv::warpAffine(_mask, _mask, M.rowRange(0,2), sz, cv::INTER_NEAREST, cv::BORDER_CONSTANT);
}

void transformer_ellipseg(cv::InputOutputArray _im, cv::InputOutputArray _mask, float p, cv::Size sz)
{
    cv::RNG rng(cv::getTickCount());
    
    /* -------------------------- Flip transformations -------------------------- */
    if (rng.uniform(0.f, 1.f) < p) // Horizontal flip
    {
        cv::flip(_im, _im, 1);
        cv::flip(_mask, _mask, 1);
    }

    if (rng.uniform(0.f, 1.f) < p) // Vertical flip
    {
        cv::flip(_im, _im, 0);
        cv::flip(_mask, _mask, 0);
    }


    /* ------------------------- Affine transformations ------------------------- */
    if (rng.uniform(0.f, 1.f) < p) // Contrast and brightness modification
    {
        float tx = rng.uniform(-10.f, 10.f);
        float ty = rng.uniform(-10.f, 10.f);
        float ang = 360*0.2*rng.uniform(-1.f, 1.f);
        float s = 1;

        affine_mask(_im, _mask, sz, tx, ty, ang, s);
    }


    /* ----------------------- Pixel value transformations ---------------------- */
    if (rng.uniform(0.f, 1.f) < p) // lighting scaling
    {
        float alpha = rng.uniform(0.05f, 2.f);
        cv::convertScaleAbs(_im, _im, alpha, 0);
    }

    if (rng.uniform(0.f, 1.f) < p) // Blur
    {
        if (rng.uniform(0.f, 1.f) < 0.5) // Motion blur
        {
            int ktype = rng.uniform(0,4);
            int ksize = (ktype == DIAG || ktype == ANTIDIAG) ? 2*rng.uniform(1,3)+1 : 2*rng.uniform(1,4)+1;

            cv::Mat kernel = getBlurKernel(ktype, ksize);
            cv::filter2D(_im, _im, -1, kernel);
        }
        else // Gaussian blur
        {
            int ksize = 2*rng.uniform(1,4)+1;
            double sigma = rng.uniform(1.,1.5);
            cv::GaussianBlur(_im, _im, {ksize,ksize}, sigma);
        }   
    }

    if (rng.uniform(0.f, 1.f) < p) // Add Gaussian noise
    {
        float stdv = rng.uniform(3.f, 12.f);
        gaussianNoise(_im, 0, stdv);
    }
}