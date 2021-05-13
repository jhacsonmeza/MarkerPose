#include "dataset.hpp"

#include "transformations.hpp" // transformer_superpoint

#include <vector>
#include <filesystem>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>



/*
-------------------- TargetData definitions --------------------
*/
TargetData::TargetData(std::filesystem::path _root, const Data& _data, bool _transform, cv::Size _sz) : root{_root},
data{_data}, transform{_transform}, sz{_sz} {}

torch::data::Example<> TargetData::get(size_t idx)
{
    cv::Mat im = cv::imread(root/data[idx].first,0);
    std::vector<float> target(data[idx].second);

    // Data augmentation
    if (transform) transformer_superpoint(im, target, 0.5, sz);


    // Image from [0,255] to [0,1]
    im.convertTo(im, CV_32FC3, 1.f/255.f);
    // Convert image to torch tensor
    auto tdata = torch::from_blob(im.data, {im.rows, im.cols, 1}, torch::kF32).clone().permute({2,0,1});


    // Final mask: {0:origin, 1:x-axis, 2:y-axis, 3:background}
    auto mask = torch::full({sz.height, sz.width}, 4, torch::kLong); // Fill with 4 for mask2labels
    for (int i = 0; i < target.size()/2; i++)
    {
        int r = cvRound(target[2*i+1]);
        int c = cvRound(target[2*i]);
        mask.index_put_({r,c}, i);
    }

    return {tdata, mask};
}

torch::optional<size_t> TargetData::size() const
{
    return data.size();
}