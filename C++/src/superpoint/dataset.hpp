#pragma once

#include <vector>
#include <string>
#include <utility> // std::pair

#include <filesystem>

#include <torch/torch.h>
#include <opencv2/core/types.hpp>

using Data = std::vector<std::pair<std::string, std::vector<float>>>;



/*
-------------------- Target data class --------------------
*/
class TargetData : public torch::data::datasets::Dataset<TargetData> {
public:
    TargetData(std::filesystem::path _root, const Data&, bool, cv::Size);
    torch::data::Example<> get(size_t);
    torch::optional<size_t> size() const;

private:
    std::filesystem::path root;
    Data data;
    bool transform;
    cv::Size sz;
};