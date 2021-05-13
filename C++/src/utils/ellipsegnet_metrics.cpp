#include "utils.hpp"

#include <tuple>
#include <vector>

#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>



// ------------------------------------------------------------------------------------------------
// ---------------------------------- EllipSegNet training utils ----------------------------------
// ------------------------------------------------------------------------------------------------
float IoU(const torch::Tensor& pred, const torch::Tensor& ref)
{
    // bitwise operations and sum over rows and cols
    auto in = torch::bitwise_and(pred,ref).sum({1,2}) + 1e-5; //(n)
    auto un = torch::bitwise_or(pred,ref).sum({1,2}) + 1e-5; //(n)
    float iou = torch::true_divide(in, un).sum().item<float>();

    return iou;
}

float centers_dist(torch::Tensor& pred, torch::Tensor& ref)
{
    float dist_sum{0.f};
    for (int i = 0; i < ref.size(0); i++)
    {
        // Reference center
        cv::Mat mask_true(ref.size(1), ref.size(2), CV_8U, ref[i].data_ptr());

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask_true, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        cv::Moments M = cv::moments(contours[0]);
        cv::Point2d c_ref(M.m10/M.m00, M.m01/M.m00);


        // Predicted center
        cv::Mat mask_pred(pred.size(1), pred.size(2), CV_8U, pred[i].data_ptr());

        contours.clear();
        contours.shrink_to_fit();
        cv::findContours(mask_pred, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        cv::Point2d c_pred(0,0);
        if (contours.size() > 0)
        {
            M = cv::moments(contours[0]);
            if (M.m00 > 0) c_pred = {M.m10/M.m00, M.m01/M.m00};
        }


        // Distance estimation
        cv::Point2d vec = c_ref - c_pred;
        double d = cv::sqrt(vec.x*vec.x + vec.y*vec.y);

        dist_sum += static_cast<float>(d);
    }

    return dist_sum;
}

std::tuple<float,float> metrics(torch::Tensor& pred, torch::Tensor& ref)
{
    ref = ref.squeeze(1) >0.5;
    pred = torch::sigmoid(pred.squeeze(1)) > 0.5;

    // Estimate intersection over union
    float iou = IoU(pred, ref);

    // Convert bool tensors to uint8 [0,255]
    ref = ref.mul(255).to(torch::kU8).cpu();
    pred = pred.mul(255).to(torch::kU8).cpu();

    // Estimate center distance
    float d = centers_dist(pred, ref);
    
    return {iou, d};
}