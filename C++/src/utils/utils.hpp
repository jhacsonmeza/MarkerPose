#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <filesystem>

#include <torch/torch.h>
#include <opencv2/core/mat.hpp>


bool compareNat(const std::string& a, const std::string& b);
std::vector<std::string> imlist(const std::filesystem::path& folder_path);


// ------------------------------------------------------------------------------------------------
// ----------------------------------- SuperPoint training utils ----------------------------------
// ------------------------------------------------------------------------------------------------
std::tuple<torch::Tensor,torch::Tensor> mask2labels(torch::Tensor& labels, int cell_size = 8);
std::tuple<float,float> metrics(torch::Tensor& det_logits, torch::Tensor& cls_logits, torch::Tensor& det_labels, torch::Tensor& cls_labels);


// ------------------------------------------------------------------------------------------------
// ---------------------------------- EllipSegNet training utils ----------------------------------
// ------------------------------------------------------------------------------------------------
float IoU(const torch::Tensor& pred, const torch::Tensor& ref);
float centers_dist(torch::Tensor& pred, torch::Tensor& ref);
std::tuple<float,float> metrics(torch::Tensor& pred, torch::Tensor& ref);



// ------------------------------------------------------------------------------------------------
// -------------------------------------- MarkerPose utilities ------------------------------------
// ------------------------------------------------------------------------------------------------
torch::Tensor labels2scores(const torch::Tensor& labels, int cell_size = 8);
torch::Tensor max_pool(const torch::Tensor& scores, int th);
torch::Tensor simple_nms(const torch::Tensor& scores, int th);

torch::Tensor sortedPoints(const torch::Tensor& scores, const torch::Tensor& labels);
void correct_patch(cv::InputArray _im, cv::OutputArray _imcrop, cv::Rect roi);
cv::Mat extractPatches(cv::InputArray _im1, cv::InputArray _im2, const torch::Tensor& kp1, const torch::Tensor& kp2, int crop_sz, int mid);
void ellipseFitting(const torch::Tensor& masks, int* pkp, int mid, cv::OutputArray _centers);

void getPose(cv::InputArray _X, cv::OutputArray R, cv::OutputArray t);
void drawAxes(cv::InputOutputArray im, cv::InputArray K, cv::InputArray dist, cv::InputArray R, cv::InputArray t);