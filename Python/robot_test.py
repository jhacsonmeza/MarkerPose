import os
import cv2
import glob
import torch
import numpy as np

from modules import utils
from modules import models


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Root path of data
root = os.path.relpath('../dataset')

# Load stereo calibration parameters
Params = cv2.FileStorage(os.path.join(root,'StereoParams.xml'), cv2.FILE_STORAGE_READ)
K1 = Params.getNode('K1').mat()
dist1 = Params.getNode('dist1').mat()


# Create SuperPoint model
superpoint = models.SuperPointNet(3)
superpoint.load_state_dict(torch.load(os.path.join(root,'py_superpoint.pt'), map_location=device))

# Create EllipSegNet model
ellipsegnet = models.EllipSegNet(16, 1)
ellipsegnet.load_state_dict(torch.load(os.path.join(root,'py_ellipsegnet.pt'), map_location=device))

# Create MarkerPose model
markerpose = models.MarkerPose(superpoint, ellipsegnet, (320,240), 120, Params)
markerpose.to(device)
markerpose.eval()


# Set window name and size
cv2.namedWindow('Pose estimaion', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Pose estimaion', 1792, 717)

# Read left and right images of the target
I1 = utils.natsort(glob.glob(os.path.join(root,'images','L','*')))
I2 = utils.natsort(glob.glob(os.path.join(root,'images','R','*')))


# Test
with torch.no_grad():
    for im1n, im2n in zip(I1,I2):
        im1 = cv2.imread(im1n)
        im2 = cv2.imread(im2n)

        # Pose estimation
        R, t = markerpose(im1, im2)

        # Visualize results
        utils.drawAxes(im1, K1, dist1, R, t)
        
        cv2.imshow('Pose estimaion',np.hstack([im1,im2]))
        if cv2.waitKey(500) == 27:
            break

    cv2.destroyAllWindows()