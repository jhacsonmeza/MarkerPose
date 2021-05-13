#include "utils.hpp"

#include <torch/torch.h>



// ------------------------------------------------------------------------------------------------
// ----------------------------------- SuperPoint training utils ----------------------------------
// ------------------------------------------------------------------------------------------------
std::tuple<torch::Tensor,torch::Tensor> mask2labels(torch::Tensor& labels, int cell_size)
{
    // label from 2D to 3D
    labels = labels.unsqueeze(1);
    auto sz = labels.sizes().vec();
    int Hc = sz[2]/cell_size, Wc = sz[3]/cell_size;

    labels = labels.view({sz[0], sz[1], Hc, cell_size, Wc, cell_size}); //(N, C, H/8, 8, W/8, 8)
    labels = labels.permute({0, 3, 5, 1, 2, 4}).contiguous(); //(N, 8, 8, C, H/8, W/8)
    labels = labels.view({sz[0], sz[1]*cell_size*cell_size, Hc, Wc});  //(N, C*8*8, H/8, W/8)

    // Add dustbin
    auto dustbin = torch::full({sz[0], 1, Hc, Wc}, 3, torch::TensorOptions().dtype(labels.dtype()).device(labels.device()));
    labels = torch::cat({labels, dustbin}, 1); //(N,C*8*8+1,H/8,W/8)

    // return classification and detection labels
    return torch::min(labels, 1); //(N,H/8,W/8)
}

std::tuple<float,float> metrics(torch::Tensor& det_logits, torch::Tensor& cls_logits, torch::Tensor& det_labels, torch::Tensor& cls_labels)
{
    det_logits = torch::argmax(det_logits,1);
    cls_logits = torch::argmax(cls_logits,1);

    float det_metric = torch::true_divide(((det_labels!=64)*(det_logits!=64)).sum({1,2}), (det_labels!=64).sum({1,2})).sum().item<float>();
    float cls_metric = torch::true_divide(((cls_labels!=3)*(cls_logits!=3)).sum({1,2}), (cls_labels!=3).sum({1,2})).sum().item<float>();
    
    return {det_metric, cls_metric};
}