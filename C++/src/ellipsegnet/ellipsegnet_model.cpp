#include "../models.hpp"

#include <torch/torch.h>


/*
-------------------- EllipSegNet definition --------------------
*/
EllipSegNetImpl::EllipSegNetImpl(int init_f, int num_outputs) : pool(torch::nn::MaxPool2dOptions(2).stride(2)), 
upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>{2,2}).mode(torch::kBilinear).align_corners(true))
{
    inc = DoubleConv(1, init_f);

    // Encoder
    down1 = DoubleConv(init_f, 2*init_f);
    down2 = DoubleConv(2*init_f, 4*init_f);
    down3 = DoubleConv(4*init_f, 4*init_f);

    // Decoder
    up1 = DoubleConv(2*4*init_f, 2*init_f, 4*init_f);
    up2 = DoubleConv(2*2*init_f, init_f, 2*init_f);
    up3 = DoubleConv(2*init_f, init_f);
    
    // Detector Head
    outc = torch::nn::Conv2d(torch::nn::Conv2dOptions(init_f, num_outputs, 1).padding(0));

    register_module("inc", inc);
    register_module("down1", down1);
    register_module("down2", down2);
    register_module("down3", down3);
    register_module("up1", up1);
    register_module("up2", up2);
    register_module("up3", up3);
    register_module("outc", outc);
}

torch::Tensor EllipSegNetImpl::forward(torch::Tensor x)
{
    auto x1 = inc(x); //(120,120)
    auto x2 = down1(pool(x1)); //(60,60)
    auto x3 = down2(pool(x2)); //(30,30)
    auto x4 = down3(pool(x3)); // (15,15)

    x = torch::cat({upsample(x4), x3}, 1); //(15*2,15*2), (30,30)
    x = up1(x);

    x = torch::cat({upsample(x), x2}, 1); //(30*2,30*2), (60,60)
    x = up2(x);

    x = torch::cat({upsample(x), x1}, 1); //(60*2,60*2), (120,120)
    x = up3(x);

    x = outc(x); // (120,120)
    
    return x;
}