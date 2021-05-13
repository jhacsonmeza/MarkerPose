import cv2
import torch
import numpy as np

from modules import utils


class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SuperPointNet(torch.nn.Module):
    def __init__(self, Nc):
        super(SuperPointNet, self).__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Shared backbone
        self.conv1 = DoubleConv(1, 64)
        self.conv2 = DoubleConv(64, 64)
        self.conv3 = DoubleConv(64, 128)
        self.conv4 = DoubleConv(128, 128)
        
        # Detector Head.
        self.convD = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        )
        
        # ID classifier Head.
        self.convC = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, Nc+1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        # Shared backbone
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        
        # Detector Head.
        det = self.convD(x)
        
        # ID classifier Head.
        cls = self.convC(x)
        
        return det, cls

class EllipSegNet(torch.nn.Module):
    def __init__(self, init_f, num_outputs):
        super(EllipSegNet, self).__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.inc = DoubleConv(1, init_f)
        self.down1 = DoubleConv(init_f, 2*init_f)
        self.down2 = DoubleConv(2*init_f, 4*init_f)
        self.down3 = DoubleConv(4*init_f, 4*init_f)
        self.up1 = DoubleConv(2*4*init_f, 2*init_f, 4*init_f)
        self.up2 = DoubleConv(2*2*init_f, init_f, 2*init_f)
        self.up3 = DoubleConv(2*init_f, init_f)
        self.outc = torch.nn.Conv2d(init_f, num_outputs, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x) #(120,120)
        x2 = self.down1(self.pool(x1)) #(60,60)
        x3 = self.down2(self.pool(x2)) #(30,30)
        x4 = self.down3(self.pool(x3)) #(15,15)
        
        x = torch.cat([self.upsample(x4), x3], 1) #(15*2,15*2), (30,30)
        x = self.up1(x)
        
        x = torch.cat([self.upsample(x), x2], 1) #(30*2,30*2), (60,60)
        x = self.up2(x)
        
        x = torch.cat([self.upsample(x), x1], 1) #(60*2,60*2), (120,120)
        x = self.up3(x)
        
        x = self.outc(x) #(120,120)
        return x


class MarkerPose(torch.nn.Module):
    def __init__(self, superpoint, ellipsegnet, imresize, crop_sz, Params):
        super(MarkerPose, self).__init__()
        self.superpoint = superpoint
        self.ellipsegnet = ellipsegnet

        self.imresize = imresize

        self.crop_sz = crop_sz
        self.mid = (crop_sz-1)//2

        # Calibration parameters
        self.K1 = Params.getNode('K1').mat()
        self.K2 = Params.getNode('K2').mat()
        self.dist1 = Params.getNode('dist1').mat()
        self.dist2 = Params.getNode('dist2').mat()

        # Create projection matrices of camera 1 and camera 2
        self.P1 = self.K1 @ np.c_[np.eye(3), np.zeros(3)]
        self.P2 = self.K2 @ np.c_[Params.getNode('R').mat(), Params.getNode('t').mat()]
    
    def pixelPoints(self, out_det, out_cls):
        scores = utils.labels2scores(out_det)
        scores = utils.simple_nms(scores, 4)
        out_cls = out_cls.argmax(1)

        kp1 = utils.sortedPoints(scores[0], out_cls[0])
        kp2 = utils.sortedPoints(scores[1], out_cls[1])

        return kp1, kp2

    def forward(self, x1, x2):
        device = next(self.parameters()).device

        # --------------------------------------------------- Pixel-level detection - SuperPoint
        # Convert image to grayscale if needed
        if x1.ndim == 3 and x2.ndim == 3:
            x1 = cv2.cvtColor(x1, cv2.COLOR_BGR2GRAY)
            x2 = cv2.cvtColor(x2, cv2.COLOR_BGR2GRAY)
        
        # Resize and stack
        imr1 = cv2.resize(x1, self.imresize) # 320x240
        imr2 = cv2.resize(x2, self.imresize) # 320x240
        imr = np.stack([imr1, imr2], 0) # 2x320x240

        # Convert images to tensor
        imt = torch.from_numpy(np.float32(imr/255)).unsqueeze(1) #2x1x320x240
        imt = imt.to(device)

        # Pixel point estimation
        out_det, out_cls = self.superpoint(imt)
        kp1, kp2 = self.pixelPoints(out_det, out_cls)
        if kp1.shape[0] < 3 or kp2.shape[0] < 3: return None, None

        # Scale points to full resolution
        s = np.array(x1.shape[::-1])/self.imresize
        kp1 = s*kp1
        kp2 = s*kp2



        # --------------------------------------------------- Ellipse segmetnation and sub-pixel center estimation - EllipSegNet
        patches = utils.extractPatches(x1, x2, kp1, kp2, self.crop_sz, self.mid)

        # Convert to tensor
        patchest = torch.from_numpy(np.float32(patches/255)).unsqueeze(1) # 2x1x120x120
        patchest = patchest.to(device)

        # Ellipse contour estimation
        out = torch.sigmoid(self.ellipsegnet(patchest))
        out = out.squeeze(1).detach().cpu().numpy()
        out = np.uint8(255*(out>0.5))

        # Ellipse fitting and sub-pixel centers estimation
        centers = utils.ellipseFitting(out)
        c1 = centers[:3] + np.int32(np.round(kp1)) - self.mid
        c2 = centers[3:] + np.int32(np.round(kp2)) - self.mid



        # --------------------------------------------------- Stereo pose estimation
        # Undistort 2D center coordinates in each image
        c1 = cv2.undistortPoints(c1.reshape(-1,1,2), self.K1, self.dist1, None, None, self.K1)
        c2 = cv2.undistortPoints(c2.reshape(-1,1,2), self.K2, self.dist2, None, None, self.K2)

        # Estimate 3D coordinates of centers through triangulation
        X = cv2.triangulatePoints(self.P1, self.P2, c1, c2)
        X = X[:3]/X[-1] # Convert from homogeneous to Euclidean

        # Marker pose estimation relative to the left camera/world frame
        Xo, Xx, Xy = X.T
        R, t = utils.getPose(Xo, Xx, Xy)

        return R, t