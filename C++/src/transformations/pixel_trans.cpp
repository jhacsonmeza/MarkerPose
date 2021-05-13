#include "transformations.hpp"

#include <cmath> // std::pow
#include <opencv2/core.hpp>


void gamma_correction(cv::InputOutputArray _im, float gamma)
{
    cv::Mat table(1, 256, CV_8U);
    uchar* ptable = table.data;
    for (int i = 0; i < 256; i++)
        ptable[i] = cv::saturate_cast<uchar>(255.f*std::pow(i/255.f, 1/gamma));
    
    cv::LUT(_im, table, _im);
}

void gaussianNoise(cv::InputOutputArray _im, float mean, float std)
{
    cv::Mat im = _im.getMat();

    // Create array with normal distributed values
    cv::setRNGSeed(cv::getTickCount());
    cv::Mat noise(im.size(), CV_32F);
    cv::randn(noise, mean, std);

    // Add noise
    uchar* pim = im.data;
    float* pnoise = noise.ptr<float>();
    for (int i = 0; i < im.total(); i++)
        pim[i] = cv::saturate_cast<uchar>(pim[i] + pnoise[i]);
}

cv::Mat getBlurKernel(int shape, int ksize)
{
    CV_Assert(shape == XDIR || shape == YDIR || shape == DIAG || shape == ANTIDIAG);

    cv::Mat kernel = cv::Mat::zeros(ksize, ksize, CV_32F);
    float* pkernel = kernel.ptr<float>();

    switch (shape)
    {
    case XDIR:
        {
            int mid = (ksize-1)/2;
            for (int i = 0; i < ksize; i++)
                pkernel[i*ksize + mid] = 1.f/ksize;
            break;
        }
    
    case YDIR:
        {
            int mid = (ksize-1)/2;
            for (int j = 0; j < ksize; j++)
                pkernel[mid*ksize + j] = 1.f/ksize;
            break;
        }
    
    case DIAG:
        for (int i = 0, j = 0; i < ksize && j < ksize; i++, j++)
            pkernel[i*ksize + j] = 1.f/ksize;
        break;
    
    case ANTIDIAG:
        for (int i = 0, j = ksize-1; i < ksize && j > -1; i++, j--)
            pkernel[i*ksize + j] = 1.f/ksize;
        break;
    }

    return kernel;
}