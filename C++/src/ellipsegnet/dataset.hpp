#pragma once

#include <vector>
#include <string>
#include <utility> // std::pair

#include <torch/torch.h>
#include <opencv2/core/types.hpp>

using Data = std::vector<std::pair<std::string, std::string>>;



/*
-------------------- Ellipse data class --------------------
*/
class EllipseData : public torch::data::datasets::Dataset<EllipseData> {
public:
    EllipseData(const Data& _data, bool _transform, cv::Size _sz);
    torch::data::Example<> get(size_t idx);
    torch::optional<size_t> size() const;

private:
    Data data;
    bool transform;
    cv::Size sz;
};