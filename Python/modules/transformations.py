import numpy as np
import cv2



# ------------------------------------------------------------------------------------------------
# ---------------------------------- Pixel value transformation ----------------------------------
# ------------------------------------------------------------------------------------------------
def gamma_correction(im, gamma):
    table = np.array([255*(i/255)**(1/gamma) for i in range(0, 256)])
    table = np.uint8(np.round(np.clip(table,0,255)))
    im = cv2.LUT(im, table)
    
    return im

def gaussianNoise(im, mean, std):
    noise = np.random.normal(mean, std, im.shape)
    im = np.uint8(np.round(np.clip(im + noise,0,255)))

    return im

def getBlurKernel(shape, ksize):
    kernel = np.zeros([ksize,ksize], np.float32)

    if shape == 0:
        kernel[(ksize-1)//2] = 1/ksize
    
    elif shape == 1:
        kernel[:,(ksize-1)//2] = 1/ksize
    
    elif shape == 2:
        np.fill_diagonal(kernel, 1/ksize)
    
    elif shape == 3:
        np.fill_diagonal(np.fliplr(kernel), 1/ksize)
    
    return kernel



# ------------------------------------------------------------------------------------------------
# ------------------------------- Points and image transformations -------------------------------
# ------------------------------------------------------------------------------------------------
def hflip(im, target):
    w = im.shape[1]
    im = cv2.flip(im,1)

    target[:,0] = w-1-target[:,0]
    
    return im, target

def vflip(im, target):
    h = im.shape[0]
    im = cv2.flip(im,0)
    
    target[:,1] = h-1-target[:,1]
    
    return im, target

def affine_pts(im, target, sz, tx, ty, theta, scale):
    # Create translation transformation
    Ht = np.array([[1.,0,tx],[0,1,ty],[0,0,1]])

    # Create rotation transformation
    T = cv2.getRotationMatrix2D(((sz[0]-1)/2,(sz[1]-1)/2), theta, 1)
    Hr = np.concatenate([T, [[0,0,1]]], 0)

    # Create scaling transformation
    T = cv2.getRotationMatrix2D(((sz[0]-1)/2,(sz[1]-1)/2), 0, scale)
    Hs = np.concatenate([T, [[0,0,1]]], 0)


    # Apply transformations
    M = Hs @ Hr @ Ht
    x, y = target.T.copy()
    target[:,0] = M[0,0]*x + M[0,1]*y + M[0,2]
    target[:,1] = M[1,0]*x + M[1,1]*y + M[1,2]

    # Bbox of the points
    bbox = cv2.boundingRect(np.float32(target))
    # Width and height of the restriction bbox
    restrict_wh = np.array([sz[0]-2*k, sz[1]-2*k])

    # width and height of union between resctriction bbox and points bbox
    k = 20
    union_xy = np.minimum([k,k],bbox[:2])
    union_wh = np.maximum([k+restrict_wh[0],k+restrict_wh[1]],[bbox[0]+bbox[2],bbox[1]+bbox[3]]) - union_xy

    if union_wh[0]*union_wh[1] > restrict_wh[0]*restrict_wh[1]:
        cx = union_wh[0] - restrict_wh[0]
        cy = union_wh[1] - restrict_wh[1]

        if union_xy[0] == k: cx = -cx
        if union_xy[1] == k: cy = -cy

        # Apply translation to target
        target[:,0] += cx
        target[:,1] += cy

        # Add translation
        Ht = np.array([[1.,0,cx],[0,1,cy],[0,0,1]])
        M = Ht @ M

    M = M[:2,:]
    im = cv2.warpAffine(im,M,None,None,cv2.INTER_LINEAR,cv2.BORDER_REPLICATE)
    
    return im, target

def transformer_superpoint(im, target, p, sz):
    ''' -------------------------- Flip transformations -------------------------- '''
    if np.random.rand() < p: im, target = hflip(im, target)
    if np.random.rand() < p: im, target = vflip(im, target)

    
    ''' ------------------------- Affine transformations ------------------------- '''
    if np.random.rand() < p:
        tx = sz[0]*0.2*np.random.uniform(-1, 1)
        ty = sz[1]*0.2*np.random.uniform(-1, 1)
        ang = 360*0.2*np.random.uniform(-1, 1)
        s = np.random.uniform(0.95, 1.2)
        
        im, target = affine_pts(im, target, sz, tx, ty, ang, s)


    ''' ----------------------- Pixel value transformations ---------------------- '''
    if np.random.rand() < p: # lighting scaling
        alfa = np.random.uniform(0.05, 2)
        im = cv2.convertScaleAbs(im, None, alfa, 0)
    
    if np.random.rand() < p: # Blur
        if np.random.rand() < 0.5: # Motion blur
            ktype = np.random.randint(0,4)
            ksize = 2*np.random.randint(1,3)+1 if ktype == 2 or ktype == 3 else 2*np.random.randint(1,4)+1

            kernel = getBlurKernel(ktype, ksize)
            im = cv2.filter2D(im, -1, kernel)

        else: # Gaussian blur
            ksize = 2*np.random.randint(1,4)+1
            sigma = np.random.uniform(1., 1.5)
            im = cv2.GaussianBlur(im, (ksize,ksize), sigma)
    
    if np.random.rand() < p: # Add Gaussian noise
        stdv = np.random.uniform(3, 12)
        im = gaussianNoise(im, 0, stdv)
    
    return im, target





# ------------------------------------------------------------------------------------------------
# -------------------------------- Mask and image transformations --------------------------------
# ------------------------------------------------------------------------------------------------
def affine_mask(im, mask, sz, tx, ty, theta, scale):
    # Create translation transformation
    Ht = np.array([[1.,0,tx],[0,1,ty],[0,0,1]])

    # Create rotation transformation
    T = cv2.getRotationMatrix2D(((sz[0]-1)/2,(sz[1]-1)/2), theta, 1)
    Hr = np.concatenate([T, [[0,0,1]]], 0)

    # Create scaling transformation
    T = cv2.getRotationMatrix2D(((sz[0]-1)/2,(sz[1]-1)/2), 0, scale)
    Hs = np.concatenate([T, [[0,0,1]]], 0)


    # Apply transformations
    M = Hs @ Hr @ Ht
    M = M[:2,:]
    
    im = cv2.warpAffine(im,M,None,None,cv2.INTER_LINEAR,cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(mask,M,None,None,cv2.INTER_NEAREST,cv2.BORDER_CONSTANT)
    
    return im, mask

def transformer_ellipseg(im, mask, p, sz):
    ''' -------------------------- Flip transformations -------------------------- '''
    # Horizontal flip
    if np.random.rand() < p:
        im = cv2.flip(im,1)
        mask = cv2.flip(mask,1)
    
    # Vertical flip
    if np.random.rand() < p:
        im = cv2.flip(im,0)
        mask = cv2.flip(mask,0)
    

    ''' ------------------------- Affine transformations ------------------------- '''
    if np.random.rand() < p:
        tx = np.random.uniform(-10, 10)
        ty = np.random.uniform(-10, 10)
        ang = 360*0.2*np.random.uniform(-1, 1)
        s = 1 #np.random.uniform(0.8, 1.2)
        
        im, mask = affine_mask(im, mask, sz, tx, ty, ang, s)
    
    
    ''' ----------------------- Pixel value transformations ---------------------- '''
    if np.random.rand() < p: # Contrast and brightness modification
        alfa = np.random.uniform(0.05, 2)
        im = cv2.convertScaleAbs(im, None, alfa, 0)
    
    if np.random.rand() < p: # Blur
        if np.random.rand() < 0.5: # Motion blur
            ktype = np.random.randint(0,4)
            ksize = 2*np.random.randint(1,3)+1 if ktype == 2 or ktype == 3 else 2*np.random.randint(1,4)+1

            kernel = getBlurKernel(ktype, ksize)
            im = cv2.filter2D(im, -1, kernel)

        else: # Gaussian blur
            ksize = 2*np.random.randint(1,4)+1
            sigma = np.random.uniform(1., 1.5)
            im = cv2.GaussianBlur(im, (ksize,ksize), sigma)
    
    if np.random.rand() < p: # Add Gaussian noise
        stdv = np.random.uniform(3, 12)
        im = gaussianNoise(im, 0, stdv)
    
    return im, mask
