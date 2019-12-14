#!/usr/bin/python

def undistort(img, mtx, dist):
    """
    Undistort image
    """
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

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