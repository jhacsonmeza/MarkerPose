#include <iostream>
#include <filesystem>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "src/models.hpp" // SuperPointNet, EllipSegNet, MarkerPose
#include "utils.hpp" // imlist, drawAxes

int main()
{
    // Defining device: CPU or GPU
    torch::DeviceType device = torch::cuda::cudnn_is_available() ? torch::kCUDA : torch::kCPU;

    // Root path of dataset
    std::filesystem::path root{"../../dataset"};

    // Load stereo calibration parameters
    cv::FileStorage Params;
    Params.open(root/"StereoParams.xml", cv::FileStorage::READ);
    cv::Mat K1, dist1;
    Params["K1"] >> K1;
    Params["dist1"] >> dist1;


    // Create SuperPoint model
    SuperPointNet superpoint(3);
    torch::load(superpoint, root/"cpp_superpoint.pt", device);

    // Create EllipSegNet model
    EllipSegNet ellipsegnet(1, 16);
    torch::load(ellipsegnet, root/"cpp_ellipsegnet.pt", device);

    // Create MarkerPose model
    MarkerPose markerpose(superpoint, ellipsegnet, cv::Size(320,240), 120, Params);
    markerpose->to(device);
    markerpose->eval();


    // Set window name and size
    cv::namedWindow("Pose estimaion", cv::WINDOW_NORMAL);
    cv::resizeWindow("Pose estimaion", 1792, 717);


    // List of images
    auto I1 = imlist(root/"images"/"L");
    auto I2 = imlist(root/"images"/"R");

    
    // Test
    torch::NoGradGuard no_grad;
    for (int i = 0; i < I1.size(); i++)
    {
        cv::Mat im1 = cv::imread(I1[i]);
        cv::Mat im2 = cv::imread(I2[i]);

        // Pose estimation
        auto [R, t] = markerpose(im1, im2);

        // Visualize results
        drawAxes(im1, K1, dist1, R, t);
        cv::Mat im;
        cv::hconcat(im1, im2, im);

        cv::imshow("Pose estimaion", im);
        if (cv::waitKey(500) == 27) break;
    }
}