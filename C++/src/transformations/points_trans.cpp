#include "transformations.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


// ------------------------------------------------------------------------------------------------
// ------------------------------- Points and image transformations -------------------------------
// ------------------------------------------------------------------------------------------------
void hflip(cv::InputOutputArray _im, cv::InputOutputArray _target)
{
    int w = _im.size().width;
    cv::flip(_im, _im, 1);

    cv::Mat target = _target.getMat(); //(1,n)
    int n = target.cols/2;
    float* ptarget = target.ptr<float>(); 
    for (int i = 0; i < n; i++)
        ptarget[2*i] = w - ptarget[2*i];
}

void vflip(cv::InputOutputArray _im, cv::InputOutputArray _target)
{
    int h = _im.size().height;
    cv::flip(_im, _im, 0);

    cv::Mat target = _target.getMat(); //(1,n)
    int n = target.cols/2;
    float* ptarget = target.ptr<float>(); 
    for (int i = 0; i < n; i++)
        ptarget[2*i+1] = h - ptarget[2*i+1];
}

void affine_pts(cv::InputOutputArray _im, cv::InputOutputArray _target, cv::Size sz, float tx, float ty, float theta, float scale)
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

    // Apply transformation to the points
    cv::Mat target = _target.getMat(); //(1,n,1)
    int n = target.cols/2;

    target = target.reshape(2, n/2); //(1,n/2,2)
    cv::transform(target, target, M.rowRange(0,2));


    // Check if points are inside the image restriction area
    cv::Rect points_bbox = cv::boundingRect(target); // Bbox of the points
    int k = 20;
    cv::Rect restriction_bbox(k, k, sz.width-2*k, sz.height-2*k);
    cv::Rect union_bbox = restriction_bbox | points_bbox; // Bbox union between points and image boxes

    if (union_bbox.area() > restriction_bbox.area())
    {
        // Estimate x and y translation to correct the points to be within the image
        int cx = union_bbox.width - restriction_bbox.width;
        int cy = union_bbox.height - restriction_bbox.height;

        // Check for translation orientation
        if (union_bbox.x == k) cx = -cx;
        if (union_bbox.y == k) cy = -cy;

        // Add translation to the homography matrix
        cv::Mat Ht = (cv::Mat_<double>(3,3) << 1, 0, cx, 0, 1, cy, 0, 0, 1);
        M = Ht*M;

        // Apply translation to target
        cv::transform(target, target, Ht.rowRange(0,2));
    }

    // Apply final transformation to the image
    cv::warpAffine(_im, _im, M.rowRange(0,2), sz, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
}

void transformer_superpoint(cv::InputOutputArray _im, cv::InputOutputArray _target, float p, cv::Size sz)
{
    cv::RNG rng(cv::getTickCount());
    
    /* -------------------------- Flip transformations -------------------------- */
    if (rng.uniform(0.f, 1.f) < p) hflip(_im, _target);
    if (rng.uniform(0.f, 1.f) < p) vflip(_im, _target);


    /* ------------------------- Affine transformations ------------------------- */
    if (rng.uniform(0.f, 1.f) < p) // Contrast and brightness modification
    {
        float tx = sz.width*0.2*rng.uniform(-1.f, 1.f);
        float ty = sz.height*0.2*rng.uniform(-1.f, 1.f);
        float ang = 360*0.2*rng.uniform(-1.f, 1.f);
        float s = rng.uniform(0.95f, 1.2f);

        affine_pts(_im, _target, sz, tx, ty, ang, s);
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