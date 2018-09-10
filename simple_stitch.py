#simple_stitching
import numpy as np
import cv2
from matplotlib.pyplot import subplot,plot,figure,show,imshow
im1=cv2.imread('1.jpg')
im2=cv2.imread('2.jpg')
h,w,c=np.shape(im1)
H,W,C=np.shape(im2)

img3=np.concatenate((im1,im2),axis=1)
img3=cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
imshow(img3)
show()
