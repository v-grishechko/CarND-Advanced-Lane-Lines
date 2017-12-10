## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/corrected_image.png "Road Transformed"
[image3]: ./examples/sobel_examples.png "Binary Example"
[image4]: ./examples/bird_eye_transform.png "Warp Example"
[image5]: ./examples/find_lines.png "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[image7]: ./examples/curvature.png "Curvature"
[image8]: ./examples/example_output.png
[video1]: ./project_video.mp4 "Video"

---

### Camera Calibration

#### 1. Briefly state how I computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first and second code cell of the IPython notebook located in `src/pipeline.ipynb`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Remove distortion from original image.

I used method from _Camera calibration_ step to correct image. Example of a distortion corrected calibration image.:
![alt text][image2]

#### 2. Create threshold binary image.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in  `src/pipelines.ipynb` at fourth cell).  Here's an example of my output for this step.  (note: also there examples of other sobel operators)

![alt text][image3]

Choosed method for creation binary image:

```python
def sobel_comb(img, ksize=3, thresh=(20, 100), thresh_color=(0, 255)):
    sobel_x = sobel(img, orient='x', sobel_kernel=ksize, thresh=thresh)
    sobel_s = sobel_hls_select(img, thresh=thresh_color)
    combined_binary = np.zeros_like(sobel_x)
    combined_binary[(sobel_s == 1) | (sobel_x == 1)] = 1
    return combined_binary

```

#### 3. Perform a perspective transform.

The code for my perspective transform includes a function called `bird_eye_transform()`, which appears in sixth cell (`src/pipepline.ipynb`.  The `bird_eye_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points, also take `width` and `height` of the image.  I chose the hardcode the source and destination points in the following manner:

```python
def bird_eye_transform(img, src, dst, width=1280, height=720):
    transform_matrix = cv2.getPerspectiveTransform(src, dst)
    transformed = cv2.warpPerspective(img, transform_matrix, (width, height), flags=cv2.INTER_LINEAR)
    return transformed

src = [[180, 720], [580, 460], [705, 460], [1120, 720]]
dst = [[260, 720], [260, 0], [1000, 0], [1000, 720]]

#example of transform
img = undist(mpimg.imread("test_images/straight_lines1.jpg"))
transformed_img = bird_eye_transform(img, np.float32(src), np.float32(dst))
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 180, 720      | 260, 720      | 
| 580, 460      | 260, 0      	|
| 705, 460     	| 1000, 0      	|
| 1120, 720     | 1000, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Identify lane-line pixels and fit their positions with a polynomial.

After applying calibration, thresholding, and a perspective transform to a road image, we have a binary image where the lane lines stand out clearly. I used method of sliding window to find pixels which can be part of road lanes and after apply polynominal function to create lines by pixels. Code of this step in `src/pipelines.ipynb` in sixth cell. 

Example of lane line finding:

![alt text][image5]

Also we can find lane-lines in next frame by values calculated in past frame (this function in `src/ipython.ipynb` in sixth cell).


#### 5. Calculate the radius of curvature of the lane and the position of the vehicle with respect to center.

To calculate the curvature of lane-line I used mathematic formula ([from this](https://www.intmath.com/applications-differentiation/8-radius-curvature.php))
I did this in 8 cell in (`src/pipeline.ipynb`):

```python
def curvature(left_fit, right_fit, leftx, rightx, ploty):
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    print(left_curverad, right_curverad)

    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    curverad = (left_curverad + right_curverad) / 2

    rightx_int = right_fit[0] * 720 ** 2 + right_fit[1] * 720 + right_fit[2]
    leftx_int = left_fit[0] * 720 ** 2 + left_fit[1] * 720 + left_fit[2]

    position = (rightx_int + leftx_int) / 2
    distance_from_center = abs((640 - position) * xm_per_pix)

    return curverad, distance_from_center, position

```

![alt text][image7]

#### 6. Final result of image with detected lines and curvature

Here is an example of my result on a test image:

![alt text][image8]

Code of drawing debug info on the image:
```python
def draw_debug_info(image, binary_warped, left_fitx, right_fitx, ploty, curv, distance_from_center, position):
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    print(right_fitx.shape)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    transform_matrix = cv2.getPerspectiveTransform(np.float32(dst), np.float32(src))
    newwarp = cv2.warpPerspective(color_warp, transform_matrix, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    result = draw_curvature(result, curv, distance_from_center, position)
    return result
```

---

### Pipeline (video)

Here's a [link to my video result](https://drive.google.com/open?id=1v1AIMhnTbxDasNkMAj7A_kz2SC03NmlN)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
The problems I encountered were almost exclusively due to lighting conditions, shadows, discoloration, etc. It wasn't difficult to dial in threshold parameters to get the pipeline to perform well on the original project video, even on the lighter-gray bridge sections that comprised the most difficult sections of the video. This would definitely be an issue in snow or in a situation where, for example, a bright white car were driving among dull white lane lines.

I've considered a few possible approaches for making my algorithm more robust. These include more dynamic thresholding (perhaps considering separate threshold parameters for different horizontal slices of the image, or dynamically selecting threshold parameters based on the resulting number of activated pixels).
Also we can assume the car lane lines can't change fast and add this prediction in alghoritm. This prediction can make car lane lines finding more precisely and drawing car lane lines on video more smoothly.


