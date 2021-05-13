import pandas as pd
import numpy as np
import torch
import os

from modules import utils
from modules.dataset import SuperPointData
from modules.models import SuperPointNet


# Defining device: CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Training on',device)



# Input data
root = os.path.abspath('data')
df = pd.read_csv(os.path.join(root,'data_centers.csv'))


# Shuffle into train and test datasets
n = len(df)
len_train = int(0.9*n)
len_test = n-len_train

df_train = df.iloc[:len_train]
df_test = df.iloc[-len_test:].reset_index(drop=True)


# Split train into validation and train datasets
n = len_train
len_train = int(0.9*n)
len_val = n-len_train

df_val = df_train.iloc[-len_val:].reset_index(drop=True)
df_train = df_train.iloc[:len_train]

print("Train dataset length:", len_train)
print("Validation dataset length:", len_val)
print("Test dataset length:", len_test)




# Training and validation dataset class
train_ds = SuperPointData(os.path.join(root,'images'), df_train, True, (320,240))
val_ds = SuperPointData(os.path.join(root,'images'), df_val, False, (320,240))

# Data loaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)


# Model
model = SuperPointNet(3)
model.to(device)

# Loss
loss_func = torch.nn.CrossEntropyLoss(reduction="mean")

# Optimizer
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

# Learning rate schedule
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.5,patience=20,verbose=1)



# Trining loop
n_epochs = 300

train_losses = []
val_losses = []
train_acc_cls = []
train_acc_det = []
val_acc_cls = []
val_acc_det = []

val_loss_min = np.Inf
for epoch in range(1, n_epochs+1):
    # Get value of the current learning rate
    current_lr = opt.param_groups[0]['lr']
    
    # keep track of training and validation loss
    train_loss = 0.0
    val_loss = 0.0

    train_metric_det, train_metric_cls = 0.0, 0.0
    val_metric_det, val_metric_cls = 0.0, 0.0

    # Train the model
    model.train()
    for xb, yb in train_dl:
        xb = xb.to(device) #(n,1,240,320)
        yb = yb.to(device) #(n,240,320)
        det_labels, cls_labels = utils.mask2labels(yb) #(n,30,40) for both

        # forward pass: compute predicted outputs by passing input to the model
        out_det, out_cls = model(xb) #(n,65,30,40) and (n,4,30,40)

        # calculate the batch losses
        loss = loss_func(out_det, det_labels) + loss_func(out_cls, cls_labels)#x10000, x0.0001

        # clear the gradients of all optimized variables
        opt.zero_grad()
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        opt.step()

        # Update train loss
        train_loss += loss.item()

        # Update metric (for acc)
        md, mc = utils.metrics_superpoint(out_det.detach(), out_cls.detach(), det_labels.detach(), cls_labels.detach())
        train_metric_det += md
        train_metric_cls += mc
    
    # Validate the model
    model.eval() # Activate dropout and BatchNorm in eval mode
    with torch.no_grad(): # Save memory bc gradients are not calculated
        for xb, yb in val_dl:
            xb = xb.to(device) #(n,1,240,320)
            yb = yb.to(device) #(n,240,320)
            det_labels, cls_labels = utils.mask2labels(yb) #(n,30,40) for both

            # forward pass: compute predicted outputs by passing input to the model
            out_det, out_cls = model(xb) #(n,65,30,40) and (n,4,30,40)

            # calculate the batch losses
            loss = loss_func(out_det, det_labels) + loss_func(out_cls, cls_labels)

            # Update validation loss
            val_loss += loss.item()

            # Update metric (for acc)
            md, mc = utils.metrics_superpoint(out_det, out_cls, det_labels, cls_labels)
            val_metric_det += md
            val_metric_cls += mc
    
    # Calculate average losses of the epoch
    train_loss /= len_train
    val_loss /= len_val
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Calculate average metrics of the epoch
    train_metric_det *= 100/len_train
    train_metric_cls *= 100/len_train
    train_acc_det.append(train_metric_det)
    train_acc_cls.append(train_metric_cls)

    val_metric_det *= 100/len_val
    val_metric_cls *= 100/len_val
    val_acc_det.append(val_metric_det)
    val_acc_cls.append(val_metric_cls)

    
    # Store best model
    if val_loss < val_loss_min:
        print(f'Validation loss decreased ({val_loss_min:.6} --> {val_loss:.6}). Saving model ...')

        torch.save(model.state_dict(), 'superpoint.pt')
        val_loss_min = val_loss
    
    # learning rate schedule
    lr_scheduler.step(val_loss)
    
    print(f"Epoch {epoch}/{n_epochs}, lr = {current_lr:.2e}, "
    f"train loss: {train_loss:.6}, val loss: {val_loss:.6}, "
    f"train det acc: {train_metric_det:.2f}%, train cls acc: {train_metric_cls:.2f}%, "
    f"val det acc: {val_metric_det:.2f}%, val cls acc: {val_metric_cls:.2f}%")
    
    print("-"*10)

np.savez('loss_metric_superpoint.npz',train_losses=np.array(train_losses),val_losses=np.array(val_losses),
train_acc_det=np.array(train_acc_det),train_acc_cls=np.array(train_acc_cls),val_acc_det=np.array(val_acc_det),
val_acc_cls=np.array(val_acc_cls))