import numpy as np
import torch

import random
import glob
import os

from modules import utils
from modules.dataset import EllipseData
from modules.models import EllipSegNet


# Defining device: CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Training on',device)


# Input data
root = os.path.abspath('data')

imlist = sorted(glob.glob(os.path.join(root,'patch120','*')))
masklist = sorted(glob.glob(os.path.join(root,'mask120','*')))
data = list(zip(imlist,masklist))

# Shuffle the data
random.seed(0)
random.shuffle(data)


# Shuffle into train and test datasets
n = len(data)
len_train = int(0.9*n)
len_test = n-len_train

data_train = data[:len_train]
data_test = data[-len_test:]

# Split train into validation and train datasets
n = len_train
len_train = int(0.9*n)
len_val = n-len_train

data_train = data_train[:len_train]
data_val = data_train[-len_val:]

print("Train dataset length:", len_train)
print("Validation dataset length:", len_val)
print("Test dataset length:", len_test)



# Train and validation datasets class
train_ds = EllipseData(data_train, True, (120,120))
val_ds = EllipseData(data_val, False, (120,120))

# Data loaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)


# Model
model = EllipSegNet(16, 1)
model.to(device)


# Loss
loss_func = torch.nn.BCEWithLogitsLoss(reduction="sum")

# Optimizer
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

# Learning rate schedule
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.1,patience=10,verbose=1)




# Trining loop
n_epochs = 150

train_losses = []
val_losses = []
train_acc = []
val_acc = []

val_loss_min = np.Inf
for epoch in range(1, n_epochs+1):
    # Get value of the current learning rate
    current_lr = opt.param_groups[0]['lr']
    
    # keep track of training and validation loss
    train_loss = 0.0
    val_loss = 0.0

    train_metric1, train_metric2 = 0.0, 0.0
    val_metric1, val_metric2 = 0.0, 0.0

    # Train the model
    model.train()
    for xb, yb in train_dl:
        xb = xb.to(device) #(n,1,120,120)
        yb = yb.to(device) #(n,1,120,120)

        # forward pass: compute predicted outputs by passing input to the model
        output = model(xb) #(n,1,120,120)

        # calculate the batch losses
        loss = loss_func(output, yb)

        # clear the gradients of all optimized variables
        opt.zero_grad()
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        opt.step()

        # Update train loss
        train_loss += loss.item()

        # Update metric (for acc)
        iou, d = utils.metrics_ellipseg(yb.detach(), output.detach())
        train_metric1 += iou
        train_metric2 += d
    
    # Validate the model
    model.eval() # Activate dropout and BatchNorm in eval mode
    with torch.no_grad(): # Save memory bc gradients are not calculated
        for xb, yb in val_dl:
            xb = xb.to(device) #(n,1,120,120)
            yb = yb.to(device) #(n,1,120,120)

            # forward pass: compute predicted outputs by passing input to the model
            output = model(xb) #(n,1,120,120)

            # calculate the batch losses
            loss = loss_func(output, yb)

            # Update validation loss
            val_loss += loss.item()

            # Update metric (for acc)
            iou, d = utils.metrics_ellipseg(yb, output)
            val_metric1 += iou
            val_metric2 += d
    
    # Calculate average losses of the epoch
    train_loss /= len_train
    val_loss /= len_val
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_metric1 *= 100/len_train
    val_metric1 *= 100/len_val
    train_metric2 /= len_train
    val_metric2 /= len_val
    train_acc.append(train_metric1)
    val_acc.append(val_metric1)

    
    # Store best model
    if val_loss < val_loss_min:
        print(f'Validation loss decreased ({val_loss_min:.6} --> {val_loss:.6}). Saving model ...')

        torch.save(model.state_dict(), 'ellipseg.pt')
        val_loss_min = val_loss
    
    # learning rate schedule
    lr_scheduler.step(val_loss)
    
    print(f"Epoch {epoch}/{n_epochs}, lr = {current_lr:.2e}, "
    f"train loss: {train_loss:.6}, val loss: {val_loss:.6}, "
    f"train acc: {train_metric1:.2f}%, val acc: {val_metric1:.2f}%, "
    f"train px err: {train_metric2:.2f} px, val px err: {val_metric2:.2f} px")
    
    print("-"*10)

np.savez('loss_metric_ellipseg.npz',train_losses=np.array(train_losses),val_losses=np.array(val_losses),
train_acc=np.array(train_acc),val_acc=np.array(val_acc))