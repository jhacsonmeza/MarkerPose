#include "utils.hpp"

#include <vector>

#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>



// ------------------------------------------------------------------------------------------------
// -------------------------------------- MarkerPose utilities ------------------------------------
// ------------------------------------------------------------------------------------------------
torch::Tensor labels2scores(const torch::Tensor& labels, int cell_size)
{
    auto scores = torch::nn::functional::softmax(labels, 1);
    scores = scores.slice(1,0,-1); // equivalent to scores[:,:-1]

    auto sz = labels.sizes().vec();
    int H = sz[2]*cell_size, W = sz[3]*cell_size;
    scores = scores.permute({0, 2, 3, 1}).reshape({sz[0], sz[2], sz[3], cell_size, cell_size});
    scores = scores.permute({0, 1, 3, 2, 4}).reshape({sz[0], H, W});

    return scores;
}

torch::Tensor max_pool(const torch::Tensor& scores, int th)
{
    return torch::nn::functional::max_pool2d(scores,
    torch::nn::functional::MaxPool2dFuncOptions(th*2+1).stride(1).padding(th));
}

torch::Tensor simple_nms(const torch::Tensor& scores, int th)
{
    auto zeros = torch::zeros_like(scores);
    auto max_mask = scores == max_pool(scores,th);

    auto supp_mask = max_pool(max_mask.to(torch::kFloat), th) > 0;
    auto supp_scores = torch::where(supp_mask, zeros, scores);
    auto new_max_mask = supp_scores == max_pool(supp_scores, th);
    max_mask = max_mask | (new_max_mask & (~supp_mask));

    return torch::where(max_mask, scores, zeros);
}



torch::Tensor sortedPoints(const torch::Tensor& scores, const torch::Tensor& labels)
{
    // Extract keypoints
    auto kp = torch::nonzero(scores > 0.015); // (n,2) with rows,cols

    // Keep the 3 keypoints with highest score
    if (kp.size(0) > 3)
    {
        auto [_, indices] = torch::topk(scores.index({kp.t()[0],kp.t()[1]}), 3, 0);
        kp = kp.index({indices});
    }

    // Class ID
    torch::Tensor rc = torch::floor_divide(kp, 8).t(); // (2,n)
    torch::Tensor id_class = labels.index({rc[0],rc[1]});

    if (torch::any(id_class == 3).item<bool>())
        id_class.index_put_({id_class == 3}, 6-id_class.sum());
    
    // Sort keypoints
    kp = kp.index({torch::argsort(id_class)});

    // from (row,col) to (x,y)
    kp = torch::fliplr(kp).to(torch::kInt32).cpu();

    // Using Shoelace formula to know orientation of point
    int* pkp = kp.data_ptr<int>();
    int A = pkp[2]*pkp[5] - pkp[4]*pkp[3] - pkp[0]*pkp[5] + pkp[4]*pkp[1] + pkp[0]*pkp[3] - pkp[2]*pkp[1];

    if (A > 0)
        kp = kp.index({torch::tensor({1,0,2})});

    return kp;
}

void correct_patch(cv::InputArray _im, cv::OutputArray _imcrop, cv::Rect roi)
{
    cv::Mat im = _im.getMat();

    cv::Rect imbox(0, 0, im.cols, im.rows);
    cv::Rect union_bbox = imbox | roi;

    // Estimate x and y translation to correct the roi to be complete size
    int cx = union_bbox.width - im.cols + 1;
    int cy = union_bbox.height - im.rows + 1;

    // Check for translation orientation
    if (union_bbox.x == 0) cx = -cx;
    if (union_bbox.y == 0) cy = -cy;

    // Translate the image
    cv::Mat Ht = (cv::Mat_<double>(2,3) << 1, 0, cx, 0, 1, cy);
    cv::warpAffine(im, im, Ht, {}, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    // Crop
    cv::Rect newroi(roi.x + cy, roi.y + cy, roi.width, roi.height);
    cv::Mat imcrop = im(newroi);
    imcrop.copyTo(_imcrop);
}

cv::Mat extractPatches(cv::InputArray _im1, cv::InputArray _im2, const torch::Tensor& kp1, const torch::Tensor& kp2, int crop_sz, int mid)
{
    cv::Mat im1 = _im1.getMat(), im2 = _im2.getMat();
    int* pkp1 = kp1.data_ptr<int>();
    int* pkp2 = kp2.data_ptr<int>();
    int n = kp1.size(0);

    std::vector<cv::Mat> patches(2*n);
    for (int i = 0; i < n; i++)
    {
        cv::Rect roi1(pkp1[2*i] - mid, pkp1[2*i+1] - mid, crop_sz, crop_sz);
        cv::Mat imcrop1 = im1(roi1).clone();

        cv::Rect roi2(pkp2[2*i] - mid, pkp2[2*i+1] - mid, crop_sz, crop_sz);
        cv::Mat imcrop2 = im2(roi2).clone();


        // Check if the patch need to be outsied of the image
        if ((imcrop1.cols != crop_sz) || (imcrop1.rows != crop_sz))
            correct_patch(im1, imcrop1, roi1);
        
        if ((imcrop2.cols != crop_sz) || (imcrop2.rows != crop_sz))
            correct_patch(im2, imcrop2, roi2);

        patches[i] = imcrop1; //0*n+i
        patches[n+i] = imcrop2; // 1*n+i
    }

    // Create Mat array with channels
    cv::Mat cropsMat;
    cv::merge(patches, cropsMat);

    return cropsMat;
}

void ellipseFitting(const torch::Tensor& masks, int* pkp, int mid, cv::OutputArray _centers)
{
    std::vector<cv::Point2f> centers(masks.size(0));
    for (int i = 0; i < masks.size(0); i++)
    {
        cv::Mat mask(masks.size(1), masks.size(2), CV_8U, masks[i].data_ptr());
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        int idx = 0;
        if (contours.size() > 1)
        {
            double max_a = 0;
            for (int j = 0; j < contours.size(); j++)
            {
                double a = contourArea(contours[j]);
                if (a > max_a)
                {
                    max_a = a;
                    idx = j;
                }
            }
        }

        cv::RotatedRect rbox = cv::fitEllipse(contours[idx]);
        centers[i] = {rbox.center.x + pkp[2*i] - mid, rbox.center.y + pkp[2*i+1] - mid};
    }

    cv::Mat(centers).copyTo(_centers);
}




void getPose(cv::InputArray _X, cv::OutputArray _R, cv::OutputArray _t)
{
    // Get rows
    cv::Mat Xo = _X.getMat(0);
    cv::Mat Xx = _X.getMat(1);
    cv::Mat Xy = _X.getMat(2);


    // Unit vector estimation
    cv::Mat xaxis = Xx-Xo; // (1,3), float
    xaxis = xaxis/cv::norm(xaxis);

    cv::Mat yaxis = Xy-Xo; // (1,3), float
    yaxis = yaxis/cv::norm(yaxis);

    cv::Mat zaxis = xaxis.cross(yaxis); // (1,3), floats


    // Build rotation matrix and translation vector
    _R.create({3,3}, CV_32F);
    cv::Mat R = _R.getMat();
    cv::Mat axis[] = {xaxis, yaxis, zaxis};
    cv::vconcat(axis, 3, R);
    R = R.t();
    
    Xo = Xo.t();
    Xo.copyTo(_t);
}

void drawAxes(cv::InputOutputArray im, cv::InputArray K, cv::InputArray dist, cv::InputArray R, cv::InputArray t)
{
    cv::Mat axes = 40*(cv::Mat_<float>(4,3) << 0,0,0, 1,0,0, 0,1,0, 0,0,1);

    // Reproject target coordinate system axes
    cv::Mat rvec;
    cv::Rodrigues(R, rvec); //(3,1,1)

    std::vector<cv::Point2f> axs;
    cv::projectPoints(axes, rvec, t, K, dist, axs);

    // Draw axes
    cv::line(im, axs[0], axs[1], {0,0,255}, 5);
    cv::line(im, axs[0], axs[2], {0,255,0}, 5);
    cv::line(im, axs[0], axs[3], {255,0,0}, 5);
}