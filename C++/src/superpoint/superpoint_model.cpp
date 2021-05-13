#include "../models.hpp"

#include <torch/torch.h>
#include <tuple>


/*
-------------------- SuperPoint definition --------------------
*/
SuperPointNetImpl::SuperPointNetImpl(int Nc) : pool(torch::nn::MaxPool2dOptions(2).stride(2))
{
    // Shared Encoder
    conv1 = DoubleConv(1, 64);
    conv2 = DoubleConv(64, 64);
    conv3 = DoubleConv(64, 128);
    conv4 = DoubleConv(128, 128);
    
    // Detector Head
    convD = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
        torch::nn::BatchNorm2d(256),
        torch::nn::ReLU(true),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 65, 1).stride(1).padding(0))
    );
    
    // ID classifier Head
    convC = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
        torch::nn::BatchNorm2d(256),
        torch::nn::ReLU(true),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, Nc+1, 1).stride(1).padding(0))
    );

    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);
    
    register_module("convD", convD);
    
    register_module("convC", convC);
}

std::tuple<torch::Tensor,torch::Tensor> SuperPointNetImpl::forward(torch::Tensor x)
{
    // Shared Encoder
    x = conv1(x);
    x = pool(x);
    x = conv2(x);
    x = pool(x);
    x = conv3(x);
    x = pool(x);
    x = conv4(x);
    
    // Detector Head
    auto det = convD->forward(x);
    
    // ID classifier Head
    auto cls = convC->forward(x);

    return {det, cls};
}