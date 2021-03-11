import numpy as np
import matplotlib.pyplot as plt
import cv2

def ImageInActualSize(I):
    dpi = plt.rcParams['figure.dpi']
    H,W = I.shape[:2]
    figSize = W/float(dpi) , H/float(dpi)
    fig = plt.figure(figsize = figSize)
    ax = fig.add_axes([0,0,1,1])
    ax.axis('off')
    ax.imshow(I,cmap='gray')
    plt.show

def resize(scale, I): # Using opencv function
    fx, fy = scale
    res = cv2.resize(I, fx=fx, fy=fy, dsize=None)
    return res

def bilinearInterpolate(r, c, I):
    lc = int(c)
    rc = lc + 1

    tr = int(r)
    br = tr + 1

    wl = rc - c
    wr = c - lc

    wt = br - r
    wb = r - tr

    if((tr>=0 and br<I.shape[0]) and (lc>=0 and rc<I.shape[1])):
        a = wl*I[tr, lc] + wr*I[tr, rc]
        b = wl*I[br, lc] + wr*I[br, rc]

        g = wt*a+wb*b

        return g
    else:
        return 0

def ScaleImageAlgo(scale, I):
    fx, fy = scale
    S = np.array([[fx, 0], [0, fy]])

    numRows = I.shape[0]
    numcols = I.shape[1]

    I2 = np.zeros((fx * numRows, fy * numcols), dtype="uint8")
    Tinv = np.linalg.inv(S)

    for i in range(I2.shape[0]):
        for j in range(I2.shape[1]):
            p_dash = np.array([i, j])
            p = Tinv.dot(p_dash)

            new_i, new_j = p
            if((new_i<0 or new_i>=numRows) or (new_j<0 or new_j>=numcols)):
                pass
            else:
                g = bilinearInterpolate(new_i, new_j, I)
                I2[i, j] = g
    return I2

# scale is a tuple consisting of width factor and height factor respectively.
# I is the numpy ndarray of image.

scale = (2, 2)
grayImage = r'../img/albert-einstein_gray.jpg'
colourImage = r'../img/tulips.jpg'
I_gray = cv2.imread(grayImage, cv2.IMREAD_GRAYSCALE)
I_BGR = cv2.imread(colourImage)

""" I_resized_gray = resize(scale, I_gray)
I_resized_BGR = resize(scale, I_BGR) """

I_resized_gray = ScaleImageAlgo(scale, I_gray)

I_col = I_BGR[:,:,::-1]
r = ScaleImageAlgo(scale, I_col[:,:,0])
g = ScaleImageAlgo(scale, I_col[:,:,1])
b = ScaleImageAlgo(scale, I_col[:,:,2])
C = np.zeros((r.shape[0], r.shape[1], 3), dtype="uint8")
C[:,:,0] = r
C[:,:,1] = g
C[:,:,2] = b

plt.figure(1)
plt.subplot(211)
plt.imshow(I_resized_gray, cmap = "gray")
plt.subplot(212)
plt.imshow(C)
plt.show()

ImageInActualSize(I_resized_gray)
plt.imsave(r'../img/resized_gray_using_algo.jpg', I_resized_gray, cmap = "gray")
ImageInActualSize(C)
plt.imsave(r'../img/resized_tulips_using_algo.jpg', C)
