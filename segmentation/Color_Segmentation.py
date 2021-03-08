import cv2
import matplotlib.pyplot as plt
import numpy as np

def color_seg(path, lr, ur):
    img = cv2.imread(path) # BGR Format

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # converting to hsv format

    mask = cv2.inRange(hsv_img, lr, ur)

    res = cv2.bitwise_and(img, img, mask = mask)

    return img, res

# path is the path of the image
# lr and ur are numpy array of lower and upper bound of the color you want to segment

path = "tulips.jpg"
lr = np.array([20,100,100]) # lower bound of yellow color
ur = np.array([30,255,255]) # upper bound of yellow color

img, res = color_seg(path, lr, ur)

plt.figure(1)

plt.subplot(121)
plt.imshow(img[:,:,::-1]) # matplotlib has RGB format

plt.subplot(122)
plt.imshow(res[:,:,::-1])
plt.imsave(r'segmented_tulips.jpg', res[:,:,::-1])

plt.show()
