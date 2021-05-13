#include "dataset.hpp"

#include "transformations.hpp" // transformer_ellipseg

#include <opencv2/opencv.hpp>
#include <torch/torch.h>



EllipseData::EllipseData(const Data& _data, bool _transform, cv::Size _sz) : data{_data}, transform{_transform}, sz{_sz} {}

torch::data::Example<> EllipseData::get(size_t idx)
{
    // Read image and mask
    cv::Mat im = cv::imread(data[idx].first,0);
    cv::Mat mask = cv::imread(data[idx].second,0);

    // Data augmentation
    if (transform) transformer_ellipseg(im, mask, 0.5, sz);
    

    // Image from [0,255] to [0,1]
    im.convertTo(im, CV_32F, 1.f/255.f);
    // Convert image to torch tensor
    auto tdata = torch::from_blob(im.data, {im.rows, im.cols, 1}, torch::kF32).clone().permute({2,0,1});

    // Mask from [0,255] to [0,1] and float32
    mask.convertTo(mask, CV_32F, 1.f/255.f);
    // Convert mask to torch tensor
    auto tmask = torch::from_blob(mask.data, {im.rows, im.cols, 1}, torch::kF32).clone().permute({2,0,1});


    return {tdata, tmask};
}

torch::optional<size_t> EllipseData::size() const
{
    return data.size();
}