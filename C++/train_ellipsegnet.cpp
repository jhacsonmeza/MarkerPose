#include <iostream>
#include <filesystem> // std::filesystem::path

#include <utility> //std::move

#include <cmath> // INFINITY
#include <algorithm> // std::shuffle
#include <random> // std::mt19937

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "src/ellipsegnet/dataset.hpp" // Data, EllipseData
#include "src/models.hpp" // EllipSegNet
#include "src/utils/utils.hpp" // metrics


int main()
{
    // Defining device: CPU or GPU
    torch::DeviceType device = torch::cuda::cudnn_is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Training on " << device << std::endl << std::endl;

    // Root path of dataset
    std::filesystem::path root{"data"};



    // List of images and masks filenames
    auto images_list = imlist(root/"patch120");
    auto masks_list = imlist(root/"mask120");

    Data data(images_list.size());
    for (int i = 0; i < images_list.size(); i++)
        data[i] = {images_list[i], masks_list[i]};
    
    std::shuffle(data.begin(), data.end(), std::mt19937(1)); // randomly shuffle the data


    // Split data into training and test
    int n = data.size();
    int len_train = int(0.9*n);
    int len_test{n-len_train};

    Data data_training(data.begin(), data.begin()+len_train);
    Data data_test(data.end()-len_test, data.end());
    
    // Split training data into training and validation
    n = len_train;
    len_train = int(0.9*n);
    int len_val{n-len_train};

    Data data_train(data_training.begin(), data_training.begin()+len_train);
    Data data_val(data_training.end()-len_val, data_training.end());

    std::cout << "Train data length: " << len_train << std::endl;
    std::cout << "Validation data length: " << len_val << std::endl;
    std::cout << "Test data length: " << len_test << std::endl << std::endl;


    // Training and validation dataset class
    auto train_ds = EllipseData(data_train, true, {120,120}).map(torch::data::transforms::Stack<>());
    auto val_ds = EllipseData(data_val, false, {120,120}).map(torch::data::transforms::Stack<>());

    // Creating data loaders
    auto train_dl = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_ds),torch::data::DataLoaderOptions(16).workers(0));
    auto val_dl = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(val_ds),torch::data::DataLoaderOptions(16).workers(0));


    // Create the model
    EllipSegNet model(16, 1);
    model->to(device);

    // Defining the loss function
    torch::nn::BCEWithLogitsLoss loss_func(torch::nn::BCEWithLogitsLossOptions().reduction(torch::kSum));

    // Defining the optimizer
    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-4));

    // Learning rate schedule
    auto& options = static_cast<torch::optim::AdamOptions&>(opt.param_groups()[0].options());
    double factor = 0.1;
    int patience = 10;


    //--------------------------------- Trining loop
    int n_epochs = 150, count_lr = 0;
    float val_loss_min = INFINITY;
    for (int epoch = 0; epoch < n_epochs; epoch++)
    {
        // Get value of the current learning rate
        double current_lr = options.lr();

        // keep track of training and validation loss
        float train_loss = 0.0, val_loss = 0.0;

        float train_metric_iou = 0.0, train_metric_dist = 0.0;
        float val_metric_iou = 0.0, val_metric_dist = 0.0;

        // Train the model
        model->train();
        for (auto& batch : *train_dl)
        {
            auto xb = batch.data.to(device); //(n,1,120,120)
            auto yb = batch.target.to(device); //(n,1,120,120)

            // Forward pass: compute predicted outputs by passing inputs to the model
            auto output = model(xb); //(n,1,120,120)

            // Calculate the batch loss
            auto loss = loss_func(output, yb);

            // Clear the gradients of all optimized variables
            opt.zero_grad();
            // Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward();
            // Perform a single optimization step (parameter update)
            opt.step();

            // Update train loss
            train_loss += loss.item<float>();

            // Update metric (for acc)
            output = output.detach();
            yb = yb.detach();
            auto [iou, d] = metrics(output, yb);
            train_metric_iou += iou;
            train_metric_dist += d;
        }

        // Validate the model
        model->eval();
        torch::NoGradGuard no_grad;
        for (auto& batch : *val_dl)
        {
            auto xb = batch.data.to(device); //(n,1,120,120)
            auto yb = batch.target.to(device); //(n,1,120,120)

            // Forward pass: compute predicted outputs by passing inputs to the model
            auto output = model(xb); //(n,1,120,120)

            // Calculate the batch loss
            auto loss = loss_func(output, yb);

            // Update validation loss
            val_loss += loss.item<float>();

            // Update metric (for acc)
            auto [iou, d] = metrics(output, yb);
            val_metric_iou += iou;
            val_metric_dist += d;
        }

        // Calculate average losses of the epoch
        train_loss /= len_train;
        val_loss /= len_val;

        // Calculate average metrics of the epoch
        train_metric_iou *= 100.f/len_train;
        train_metric_dist /= len_train;

        val_metric_iou *= 100.f/len_val;
        val_metric_dist /= len_val;


        // Store best model - learning rate schedule
        if (val_loss < val_loss_min)
        {
            std::cout << "Validation loss decreased (" << val_loss_min << " --> " << val_loss << "). Saving model..." << std::endl;
            torch::save(model, "../ellipsegnet.pt");

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
        << ", train acc: " << train_metric_iou << "% , val acc: " << val_metric_iou
        << "%, train px err: " << train_metric_dist << ", val px err: " << val_metric_dist << std::endl;

        std::cout << "----------" << std::endl;
    }
}