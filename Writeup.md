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

This project is made of three parts:

1. Camera calibration, where the camera matrix and distortion coefficients have been computed
2. Images processing, where the pipeline for the video is beforehand implemented on single frames
3. Video stream processing, where the pipeline is actually implemented

Please be aware that the code for the images has been refactored to better suits the purpose of the video pipeline.

## Camera calibration

The projects starts off with the camera calibration, necessary to correct for distortions introduced by the camera lenses. They affect the possibility to extract correct measurements from the image so they must be compensated.

With 17 images of a chessboard (from different orientation) and the OpenCv Api function ```cv2.```, the distortion coefficient and the camera matrix of the camera has been computed. The result is clearly visible here:

![distcomp]

## Image processing

### 1. Correct to get undistorted image

The parameters previously obtained in the camera calibration have been applied here. Even in this case we can see the result (e.g. by looking at the white car on the left)

![undist]

### 2. Color and gradient filters

This step is of fundamental importantance because it is where everything that does not belong to a lane mark must be filtered out. In order to filter the shadows 2 Sobel filters have to be used (in the x and in the y directons), combined with a filter on the saturation. Here the functions that implement that:

```python
def color_filter(img, low_thresh, high_thresh):
    color_binary = np.zeros_like(img)
    color_binary[(img >= low_thresh) & (img <= high_thresh)] = 1
    return color_binary
```
```python
def sobel_filter(img, mode='x', low_thresh=0, high_thresh=255, kernel=3):
    """
    Sobel filter implementation. can be on the gradient in the x direction, y
    y direction or gradient magnitude
    """
    
    # Select mode between gradient in x or y, magnitude and direction
    if mode in ['x','y','mag']:
        
        x_flag = int(mode == 'x')
        y_flag = int(mode == 'y')  
        if x_flag or y_flag:
            sobel_out = cv2.Sobel(img, cv2.CV_64F, x_flag, y_flag, ksize=kernel)
            abs_sobel = np.abs(sobel_out)
        else:
            sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)
            sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)
            abs_sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        
        # Threshold absolute gradient
        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= low_thresh) & (scaled_sobel <= high_thresh)] = 1
        
    else:
        print("Sobel param 'mode' must correnspond to one of the following values: 'x','y','mag','dir'")
        sys.exit(-1)
    
    return sobel_binary
```
In the code it is present another functions that allows the user to combine different filters by passing for each one a dictionary with their parameters and a function that combines the filter.

Here below the image where the filter has been applied. 

![cgf]

### 3. Perspective transform

The perspective transform mathematically corrensponds to a linear transform from trapezoid in the original image to a rectangle in the warped image, in order to get a bird view of the road. 

The vertices of the trapezoid have been chosen so that the bottom base is as wide as the road and the two sides are roughly parallel to lane lines in an image of a straight road. This is aimed at achieving a correct transformation that actually provides a correct road view.

From the code point of view, it is simply a call to an OpenCv function, once provided the transformation matrix, that can be computed also with the same API.

In the code the vertices have been parametrized with the respect to the image shape, so that they can be easily adjusted.

```python

imshape = combi.shape # [y, x]
trap_topwidth = 0.085
trap_bottomwidth = .60
trap_height = 0.37
warp_width = .5
bottom_offset = 0.08
# src and s
src = np.float32([[imshape[1] * (1 - trap_bottomwidth)/2,imshape[0]*(1 - bottom_offset)],
                  [imshape[1] * (1 - trap_topwidth)/2, imshape[0] * (1 - trap_height)],
                  [imshape[1] * (1 + trap_topwidth)/2, imshape[0] * (1 - trap_height)],
                  [imshape[1] * (1 + trap_bottomwidth)/2, imshape[0]*(1 - bottom_offset)]])
dst = np.float32([[imshape[1] * (1 - warp_width)/2, imshape[0]*(1 - bottom_offset)],
                  [imshape[1] * (1 - warp_width)/2, 0],
                  [imshape[1] * (1 + warp_width)/2, 0],
                  [imshape[1] * (1 + warp_width)/2, imshape[0]*(1 - bottom_offset)]])
```
Here the functions that compute the matrix and transform the image

```python
def perspective_transform(img, M, transf=cv2.INTER_LINEAR):
    warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]), flags=transf)
    return warped
```

```python
M = cv2.getPerspectiveTransform(src, dst)
warped_img = perspective_transform(combi, M)
```
Here the perspective transform regions on both sides:

![perspective]

### 4. Lane Pixels Fit

In order to fit the lane pixels with two 2nd order polynomials, they have to be identified in the binary warped image.

The sliding window technique has been here adopted. It works by dividing the images in horizontal slices and by iterating slice by slice with a window of a certain marign. The window recenters itself on the mean position of the pixels found in the window. The windows (one for each lane line) start by computing an histogram on the bottom half of the image.

![sliding]

The other possibility, available only when previous data are available (i.e. in the video stream), simply consists of looking for lane pixels around the area of the previous polynomial, within a certain margin specified by the user.

![prior]

They have been implemented by the methods ```sliding_window``` and ```search_around_poly``` belonging to the class ```Processor```, available in the ```lanetracker``` module.

### 5. Measure radius of curvature and offset

They can be computed simply by the polynomial coefficient.

```python
def measure_curvature_real(left_fit_cr, right_fit_cr, bottom_offset, imshape,ym_per_pix = 30/720, xm_per_pix = 3.7/700):
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = imshape[0] * (1-bottom_offset)  * ym_per_pix
    
    # Implement the calculation of R_curve (radius of curvature)
    left_curverad = (1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**(3/2) / np.abs(2*left_fit_cr[0])
    right_curverad =(1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**(3/2) / np.abs(2*right_fit_cr[0])
    
    return left_curverad, right_curverad
```

```python
def measure_offset(left_fit,right_fit, dst, imshape, xm_per_pix = 3.7/700):
    y_eval = dst[3,1]
    left_point = np.polyval(left_fit, y_eval)
    right_point = np.polyval(right_fit, y_eval)
    lane_center_pos = (right_point + left_point)//2
    offset_pix = lane_center_pos - imshape[1]//2
    offset  = offset_pix * xm_per_pix
    return offset
```

### 6. Draw lines

To apply the polynomial on the original image, the inverse transformation can be applied and the lane drawn onto the image:

![final]

## Video pipeline

The code has undergone a thorough refactoring to apply all the steps on each frame of the video. 

The method ```process_frame``` starts by applying the sliding window on the first frame and then always tries to search around the previous polynomial, going back to the sliding window search when unsuccessful. If even in this case no pixels can be found, it uses the previous line.

The outliers are identified by looking at variation of the lane width and the difference in the radius of curvature. When they rise above certain thresholds, the measurement is reported as outlier (it is treated by the program as an unsuccesful search).

In order to smooth the signal, an average filter has been applied both on the polynomial coefficients and on the radius of curvature. The offset instead does not show as much noise, so its values are just considered reliable and stored as they are.

<video width="480" height="270" controls>
  <source src="output_videos/video_output.mp4" type="video/mp4">
</video>

## Discussion

Even if it performs well on the video here showed, it may face other situation where it fails to detect the lane.

The most critical part of the pipeline is undoubtedly the color and gradient trasformation: even with the best algorithm for fitting and smoothing the line, without having correctly identified the pixels, it is impossible to have an accurate precision. This pipeline may fail if there is something on the roads that tricks the filter into considering it as lane marks.

The value of the curvature, yet being realistic as computed in the pipeline, depends too much on the polynomial line, so that a slight variation in the polynomial coefficients entails a large variation in the radius of curvature.

A way to improve this project would be to find a way to explore new filters and different combinations of them, so as to improve robustness on harder scenarios, especially when they persist (single outlier rejection is not such a issue).
 




 
