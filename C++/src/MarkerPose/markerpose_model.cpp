#include "../models.hpp"

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "utils.hpp"


/*
-------------------- MarkerPose definition --------------------
*/
MarkerPoseImpl::MarkerPoseImpl(SuperPointNet _superpoint, EllipSegNet _ellipsegnet, cv::Size _imresize, int _crop_sz,
const cv::FileStorage& Params) : superpoint{_superpoint}, ellipsegnet{_ellipsegnet}, imresize{_imresize}, crop_sz{_crop_sz}
{
    mid = (crop_sz-1)/2;

    // Calibration parameters
    cv::Mat R, t;
    Params["K1"] >> K1;
    Params["K2"] >> K2;
    Params["R"] >> R;
    Params["t"] >> t;
    Params["dist1"] >> dist1;
    Params["dist2"] >> dist2;

    // Create projection matrices of camera 1 and camera 2
    hconcat(K1, cv::Mat::zeros(3,1,CV_64F), P1);
    hconcat(K2*R, K2*t, P2);

    // Register SuperPoint and EllipSegNet
    register_module("superpoint", superpoint);
    register_module("ellipsegnet", ellipsegnet);
}

std::tuple<torch::Tensor,torch::Tensor> MarkerPoseImpl::pixelPoints(torch::Tensor& out_det, torch::Tensor& out_cls)
{
    auto scores = labels2scores(out_det);
    scores = simple_nms(scores, 4);
    out_cls = out_cls.argmax(1);

    auto kp1 = sortedPoints(scores[0], out_cls[0]);
    auto kp2 = sortedPoints(scores[1], out_cls[1]);

    return {kp1, kp2};
}

std::tuple<cv::Mat,cv::Mat> MarkerPoseImpl::forward(cv::InputArray _x1, cv::InputArray _x2)
{
    auto device = superpoint->parameters()[0].device();

    // --------------------------------------------------- Pixel-level detection - SuperPoint
    // Convert image to grayscale if needed
    cv::Mat x1, x2;
    if (_x1.channels() == 3 || _x2.channels() == 3)
    {
        cv::cvtColor(_x1, x1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(_x2, x2, cv::COLOR_BGR2GRAY);
    }
    else
    {
        x1 = _x1.getMat();
        x2 = _x2.getMat();
    }

    // Resize and stack
    cv::Mat imr, imr1, imr2;
    cv::resize(x1, imr1, imresize);
    cv::resize(x2, imr2, imresize);
    std::vector<cv::Mat> ch{imr1, imr2};
    cv::merge(ch, imr);

    // Convert image to tensor
    torch::Tensor imt = torch::from_blob(imr.data, {imr.rows, imr.cols, imr.channels()}, torch::kU8); //320x240x2
    imt = imt.to(torch::kF32).mul(1.f/255.f); // From U8 to F32
    imt = imt.permute({2,0,1}).unsqueeze(1).to(device); // 2x1x320x240

    // Pixel point estimation
    auto [out_det, out_cls] = superpoint(imt);
    auto [kp1, kp2] = pixelPoints(out_det, out_cls);
    
    // Scale points to full resolution
    int* pkp1 = kp1.data_ptr<int>();
    int* pkp2 = kp2.data_ptr<int>();
    float sx = static_cast<float>(x1.cols)/imresize.width, sy = static_cast<float>(x1.rows)/imresize.height;
    for (int i = 0; i < kp1.size(0); i++)
    {
        pkp1[2*i] = cvRound(pkp1[2*i]*sx);
        pkp1[2*i+1] = cvRound(pkp1[2*i+1]*sy);

        pkp2[2*i] = cvRound(pkp2[2*i]*sx);
        pkp2[2*i+1] = cvRound(pkp2[2*i+1]*sy);
    }
    

    
    // --------------------------------------------------- Ellipse segmetnation and sub-pixel center estimation - EllipSegNet
    // Get patches
    cv::Mat patches = extractPatches(x1, x2, kp1, kp2, crop_sz, mid);

    // Convert Mat to Tensor
    torch::Tensor patchest = torch::from_blob(patches.data, {crop_sz, crop_sz, patches.channels()}, torch::kU8);
    patchest = patchest.to(torch::kF32).mul(1.f/255.f); // From U8 to F32
    patchest = patchest.permute({2,0,1}).unsqueeze(1).to(device);

    // Circle contour estimation
    torch::Tensor out = torch::sigmoid(ellipsegnet(patchest));
    out = out.squeeze(1).detach().cpu();
    out = (out > 0.5).mul(255).to(torch::kU8);

    // Ellipse center estimation
    std::vector<cv::Point2f> c1, c2;
    ellipseFitting(out.index({torch::indexing::Slice(0, 3)}), pkp1, mid, c1);
    ellipseFitting(out.index({torch::indexing::Slice(3, 6)}), pkp2, mid, c2);

    

    // --------------------------------------------------- Stereo pose estimation
    // Undistort 2D center coordinates in each image
    cv::undistortPoints(c1, c1, K1, dist1, {}, K1);
    cv::undistortPoints(c2, c2, K2, dist2, {}, K2);

    // Estimate 3D coordinate of the concentric circles through triangulation
    cv::Mat X;
    cv::triangulatePoints(P1, P2, c1, c2, X); // (4,3,1) - last element is channels
    convertPointsFromHomogeneous(X.t(), X); // (3,1,3)
    X = X.reshape(1); // (3,3,1) - each row is a point

    // Marker pose estimation relative to the left camera/world frame
    cv::Mat R, t;
    getPose(X, R, t);

    return {R, t};
}