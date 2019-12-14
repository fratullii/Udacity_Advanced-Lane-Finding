# Writeup
# Advanced Lane Finding Project

[//]: # (Image References)

[cgf]: ./output_images/cg_filters.jpg
[final]: ./output_images/lines_drawn.jpg
[perspective]: ./output_images/perspective_transform.jpg
[prior]: ./output_images/search_prior.jpg
[sliding]: ./output_images/sliding_window.jpg
[undist]: ./output_images/undist_road.jpg
[distcomp]: ./output_images/distortion_comparison.jpg

## Camera calibration

The projects starts off with the camera calibration, necessary to correct for distortions introduced by the camera lenses. They affect the possibility to extract correct measurements from the image so they must be compensated.

With 17 images of a chessboard (from different orientation) and the OpenCv Api function ```cv2.```, the distortion coefficient and the camera matrix of the camera has been computed. The result is clearly visible here:

![distcomp]

## Image processing

### 1. Correct to get undistorted image

The parameters previously obtained in the camera calibration have been applied here. Even in this case we can see the result (e.g. by looking at the white car on the left)

![undist]

### 2. Color and gradient filters

This step is of fundamental importantance because it is where everything that does not belong to a lane mark must be filtered out. So at the 	

## Video pipeline

 