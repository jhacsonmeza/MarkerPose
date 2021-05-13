from modules.transformations import transformer_superpoint, transformer_ellipseg

import os
import cv2
import torch
import numpy as np



class SuperPointData(torch.utils.data.Dataset):
    def __init__(self, root, df, transform=None, size=None):
        self.root = root
        self.imlist = df.FileName
        self.target = df[['c0x','c0y','cXx','cXy','cYx','cYy']].values
        
        # Transformation for data augmentation
        self.transform = transform
        self.size = size # w,h
    
    def __len__(self):
        return len(self.imlist)
    
    def __getitem__(self, idx):
        im = cv2.imread(os.path.join(self.root,self.imlist[idx]),0)
        target = self.target[idx].copy().reshape(-1,2)
        
        if self.transform:
            im, target = transformer_superpoint(im, target, 0.5, self.size)
        
        # From np.array (HxWxC) to torch.tensor (CxHxW). From [0,255] to [0,1]
        im = torch.from_numpy(np.float32(im/255)).unsqueeze(0)

        # Final mask: {0:origin, 1:x-axis, 2:y-axis, 3:background}
        target = np.int32(np.round(target))
        mask = torch.full(self.size[::-1], 4, dtype=torch.long)
        mask[target[:,1],target[:,0]] = torch.tensor([0,1,2],dtype=torch.long)
        
        return im, mask


class EllipseData(torch.utils.data.Dataset):
    def __init__(self, data, transform, size):
        self.data = data
        self.transform = transform
        self.size = size # w,h
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Read image and mask
        im = cv2.imread(self.data[idx][0],0)
        mask = cv2.imread(self.data[idx][1],0)
        
        if self.transform:
            im, mask = transformer_ellipseg(im, mask, 0.5, self.size)
        
        # From np.array (HxWxC) to torch.tensor (CxHxW). From [0,255] to [0,1]
        im = torch.from_numpy(np.float32(im/255)).unsqueeze(0)
        mask = torch.from_numpy(np.float32(mask/255)).unsqueeze(0)
        
        return im, mask