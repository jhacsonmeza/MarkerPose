#include <iostream>

#include <filesystem> // std::filesystem::path
#include <fstream> // std::ifstream, std::getline
#include <sstream> // std::stringstream
#include <string>

#include <vector>
#include <utility> // std::move

#include <cmath> // INFINITY

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "src/superpoint/dataset.hpp" // Data, TargetData
#include "src/models.hpp" // SuperPointNet
#include "src/utils/utils.hpp" // mask2labels, metrics


int main()
{
    // Defining device: CPU or GPU
    torch::DeviceType device = torch::cuda::cudnn_is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Training on " << device << std::endl << std::endl;

    // Root path of dataset
    std::filesystem::path root{"data"};



    // Load csv data
    std::ifstream file(root/"data_centers.csv"); // Stream to the input csv file

    // Read first line and ignore (title)
    std::string line;
    std::getline(file, line); //file >> line; split in white spaces
    
    Data data;
    while (std::getline(file, line)) // Load data
    {
        std::stringstream ss(line);

        std::string filename;
        std::getline(ss,filename,',');
        
        std::vector<float> labels(6);
        for (int i = 0; i < 6; i++)
        {
            // Save xc, yc, width, height values
            std::string label;
            std::getline(ss,label,',');
            labels[i] = std::stof(label);
        }
        
        data.push_back({filename, labels});
    }

    // Split data into training and validation datasets
    int n = data.size();
    int len_train = int(0.9*n);
    int len_val{n-len_train};

    Data data_train(data.begin(), data.begin()+len_train);
    Data data_val(data.end()-len_val, data.end());

    std::cout << "Training data length: " << len_train << std::endl;
    std::cout << "Validation data length: " << len_val << std::endl << std::endl;


    // Training and validation dataset class
    auto train_ds = TargetData(root/"images", data_train, true, {320,240}).map(torch::data::transforms::Stack<>());
    auto val_ds = TargetData(root/"images", data_val, false, {320,240}).map(torch::data::transforms::Stack<>());

    // Creating data loaders
    auto train_dl = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_ds),torch::data::DataLoaderOptions(16).workers(0));
    auto val_dl = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(val_ds),torch::data::DataLoaderOptions(16).workers(0));


    // Create the model
    SuperPointNet model(3);
    model->to(device);

    // Defining the loss function
    torch::nn::CrossEntropyLoss loss_func(torch::nn::CrossEntropyLossOptions().reduction(torch::kMean));

    // Defining the optimizer
    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-4));

    // Learning rate schedule
    auto& options = static_cast<torch::optim::AdamOptions&>(opt.param_groups()[0].options());
    double factor = 0.5;
    int patience = 20;


    //--------------------------------- Trining loop
    int n_epochs = 300, count_lr = 0;
    float val_loss_min = INFINITY;
    for (int epoch = 0; epoch < n_epochs; epoch++)
    {
        // Get value of the current learning rate
        double current_lr = options.lr();

        // keep track of training and validation loss
        float train_loss = 0.0, val_loss = 0.0;

        float train_metric_det = 0.0, train_metric_cls = 0.0;
        float val_metric_det = 0.0, val_metric_cls = 0.0;

        // Train the model
        model->train();
        for (auto& batch : *train_dl)
        {
            auto xb = batch.data.to(device); //(n,1,240,320)
            auto yb = batch.target.to(device); //(n,240,320)
            auto [cls_labels, det_labels] = mask2labels(yb); //(n,30,40) for both

            // Forward pass: compute predicted outputs by passing inputs to the model
            auto [out_det, out_cls] = model(xb); //(n,65,30,40) and (n,4,30,40)
            // Calculate the batch loss
            auto loss = loss_func(out_det, det_labels) + loss_func(out_cls, cls_labels);

            // Clear the gradients of all optimized variables
            opt.zero_grad();
            // Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward();
            // Perform a single optimization step (parameter update)
            opt.step();

            // Update train loss
            train_loss += loss.item<float>();

            // Update metrics
            out_det = out_det.detach();
            out_cls = out_cls.detach();
            det_labels = det_labels.detach();
            cls_labels = cls_labels.detach();
            auto [md, mc] = metrics(out_det, out_cls, det_labels, cls_labels);
            train_metric_det += md;
            train_metric_cls += mc;
        }

        // Validate the model
        model->eval();
        torch::NoGradGuard no_grad;
        for (auto& batch : *val_dl)
        {
            auto xb = batch.data.to(device); //(n,1,240,320)
            auto yb = batch.target.to(device); //(n,240,320)
            auto [cls_labels, det_labels] = mask2labels(yb); //(n,30,40) for both

            // Forward pass: compute predicted outputs by passing inputs to the model
            auto [out_det, out_cls] = model(xb); //(n,65,30,40) and (n,4,30,40)
            // Calculate the batch loss
            auto loss = loss_func(out_det, det_labels) + loss_func(out_cls, cls_labels);

            // Update validation loss
            val_loss += loss.item<float>();

            // Update metrics
            auto [md, mc] = metrics(out_det, out_cls, det_labels, cls_labels);
            val_metric_det += md;
            val_metric_cls += mc;
        }

        // Calculate average losses of the epoch
        train_loss /= len_train;
        val_loss /= len_val;

        // Calculate average metrics of the epoch
        train_metric_det *= 100.f/len_train;
        train_metric_cls *= 100.f/len_train;

        val_metric_det *= 100.f/len_val;
        val_metric_cls *= 100.f/len_val;


        // Store best model - learning rate schedule
        if (val_loss < val_loss_min)
        {
            std::cout << "Validation loss decreased (" << val_loss_min << " --> " << val_loss << "). Saving model..." << std::endl;
            torch::save(model, "../superpoint.pt");

            val_loss_min = val_loss;
            count_lr = 0; // Reset counter
        }
        else if (++count_lr > patience)
        {
            std::cout << "Reducing learning rate to " << current_lr*factor <<  std::endl;
            options.lr(current_lr*factor);

            count_lr = 0; // Reset counter
        }


        std::cout << "Epoch " << epoch+1 << "/" << n_epochs << ", "
        << "lr = " << current_lr << ", "
        << "train loss: " << train_loss << ", val loss: " << val_loss
        << ", train det acc: " << train_metric_det << "% , train cls acc: " << train_metric_cls 
        << "%, val det acc: " << val_metric_det << "% , val cls acc: " << val_metric_cls << "%" << std::endl;

        std::cout << "----------" << std::endl;
    }
}