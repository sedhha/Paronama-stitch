# Paronama-stitch
3 ways of stitching images to create paranoma:c1. Direct stitching 2. The scale-invariant feature transform 3. Speeded up Robust feature transform.
Images are provided in the folder.
There are two python files, one is based on feature recognition, the other is direct side by side stitching.
In first program to switch from sift to surf you just need to change 
    kp2, des2 = sift.detectAndCompute(img2,None)
with
    kp2, des2 = surf.detectAndCompute(img2,None)
And you are good to go.
