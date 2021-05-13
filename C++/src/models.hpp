#pragma once

#include <opencv2/core.hpp>
#include <torch/data.h>
#include <torch/nn.h>
#include <tuple>


class DoubleConvImpl : public torch::nn::Module {
public:
    DoubleConvImpl() {}

    DoubleConvImpl(int in_channels, int out_channels, int mid_channels = 0)
    {
        if (mid_channels == 0)
            mid_channels = out_channels;
        
        double_conv = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, mid_channels, 3).padding(1)),
            torch::nn::BatchNorm2d(mid_channels),
            torch::nn::ReLU(true),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, out_channels, 3).padding(1)),
            torch::nn::BatchNorm2d(out_channels),
            torch::nn::ReLU(true)
        );

        register_module("double_conv", double_conv);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        return double_conv->forward(x);
    }

private:
    torch::nn::Sequential double_conv = nullptr;
};
TORCH_MODULE(DoubleConv);



class SuperPointNetImpl : public torch::nn::Module {
public:
    SuperPointNetImpl(int Nc);
    std::tuple<torch::Tensor,torch::Tensor> forward(torch::Tensor x);

private:
    torch::nn::MaxPool2d pool;
    
    // Shared Encoder
    DoubleConv conv1, conv2, conv3, conv4;
    
    // Detector Head
    torch::nn::Sequential convD = nullptr;
    
    // ID classifier Head
    torch::nn::Sequential convC = nullptr;
};
TORCH_MODULE(SuperPointNet);



class EllipSegNetImpl : public torch::nn::Module {
public:
    EllipSegNetImpl(int init_f, int num_outputs);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::MaxPool2d pool;
    torch::nn::Upsample upsample;

    // Encoder
    DoubleConv inc, down1, down2, down3;

    // Decoder
    DoubleConv up1, up2, up3;
    torch::nn::Conv2d outc = nullptr;
};
TORCH_MODULE(EllipSegNet);



class MarkerPoseImpl : public torch::nn::Module {
public:
    MarkerPoseImpl(SuperPointNet _superpoint, EllipSegNet _ellipsegnet, cv::Size _imresize, int _crop_sz, const cv::FileStorage& Params);
    std::tuple<torch::Tensor,torch::Tensor> pixelPoints(torch::Tensor& out_det, torch::Tensor& out_cls);
    std::tuple<cv::Mat,cv::Mat> forward(cv::InputArray _x1, cv::InputArray _x2);

private:
    int crop_sz, mid;
    cv::Size imresize;
    cv::Mat K1, K2, dist1, dist2, P1, P2;

    SuperPointNet superpoint;
    EllipSegNet ellipsegnet;
};
TORCH_MODULE(MarkerPose);