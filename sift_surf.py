#today's final attempt

#Rough
import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread('xx.jpg',0)          # queryImage
img2 = cv2.imread('xy.jpg',0) # trainImage

# Initiate SIFT detector
sift=cv2.xfeatures2d.SIFT_create()
surf=cv2.xfeatures2d.SURF_create()
orb=cv2.ORB_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
#matches = bf.match(des1,des2)
matches = flann.knnMatch(des1,des2,k=2)
good=[]
# store all the good matches as per Lowe's ratio test.
#matches = sorted(matches, key = lambda x:x.distance)
#good=matches
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None
row1,col1=np.shape(img1)
row2,col2=np.shape(img2)
#reference: https://github.com/opencv/opencv/issues/6072
'''draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)'''
end1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
#end2 = tuple(np.round(kp2[m.queryIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
end2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) ) #np.array([img1.shape[1], 0]))

ic1=np.delete(img1,slice(end1[1],col1),axis=1)
ic2=np.delete(img2,slice(0,end2[1]),axis=1)
ixf=np.concatenate((ic1,ic2),axis=1)
