# Getting Started

Install and import the OpenCV library.

```py
pip3 install opencv-python
```

```py
import cv2
```

# Image Manipulation

## Reading and Displaying an Image

```py
img = cv2.imread('cute_cat.jpeg')
cv2.imshow('Cute Cat', img)
cv2.waitKey(0)
```

<details><summary><strong>cv2.imread(path, flag)</strong></summary>

<br>

Capture an image from a specified file, which you can then assign to a variable. If an image can't be read, this method returns an empty matrix.

This method accepts two parameters:

- `path`: the path to the specified image in a string format.
- `flag`: specifies the way in which the image should be read. The default flag is `cv2.IMREAD_COLOR`.

These are the three possible flag parameters for the method:

- `cv2.IMREAD_COLOR` (default): It specifies to load a color image. Any transparency of image will be neglected. Alternatively, we can pass integer value 1 for this flag.
- `cv2.IMREAD_GRAYSCALE`: It specifies to load an image in grayscale mode. Alternatively, we can pass integer value 0 for this flag.
- `cv2.IMREAD_UNCHANGED`: It specifies to load an image as such including alpha channel. Alternatively, we can pass integer value -1 for this flag.

</details>

<details><summary><strong>cv2.imshow(window_name, image)</strong></summary>

<br>

Display an image in a new window. The window will automatically scale to the image size.

This method accepts two parameters:

- `window_name`: a string respresenting the name of the window in which the image will be displayed.
- `image`: the source image that will be displayed in the window.

</details>

<details><summary><strong>cv2.waitKey(delay)</strong></summary>

<br>

Necessary to avoid the script from immediately terminating.

The method accepts a `delay` input in milliseconds. This is the time that the script will wait for the program to continue. If `0` is passed, the program will wait for input indefinitely. In this case, if waitkey is not used, the program will automatically terminate after the imshow line runs, resulting in the image flashing in screen for a fraction of a second. Passing a `0` wil ensure the image stays on screen until the user chooses to close it.

</details>

<br>

## Reading and Displaying a Video

```py
video = cv.VideoCapture('cute_dog.mp4')
while True:
    isTrue, frame = video.read()
    cv2.imshow('Cute Dog', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
video.release()
cv2.destroyAllWindows()

```

<details><summary><strong>cv2.VideoCapture(path/source)</strong></summary>

<br>

Create a video capture object from a source, which can then be stored in a variable.

This method accepts the source of the video as a parameter. Passing path to a video as a string will allow you to use a local video file. An integer can also be passed, and refers to a camera on the computer. Passing `0` will typically capture video from a webcam. Passing subsequent integers will allow accessing other cameras.

</details>

<details><summary><strong>video.read()</strong></summary>

<br>

Read the video frame by frame. It returns a boolean that tells us whether reading the frame was successful, and the frame itself. The operation needs to be performed inside a while loop.

</details>

<details><summary><strong>video.release()</strong></summary>

<br>

Close video or capturing device. Must be called before creating another instance of the video capture object.

</details>

<details><summary><strong>cv2.destroyAllWindows()</strong></summary>

<br>

Destroy all currently open windows. To destroy a specific window, use the function `cv2.destroyWindow()` where you pass the exact window name.

</details>

<br>

## Resizing and Rescaling an Image

```py
img = cv2.imread('cute_cat.jpeg')

def rescaleFrame(image, scale=0.75):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

cv2.imshow('Cute Cat', rescaleFrame(img))
cv2.waitKey(0)
```

<details><summary><strong>frame.shape</strong></summary>

<br>

The `shape` property of an image returns the following tuple: (height, width, num_of_channels). For example a colored image with a resolution of 1920x1080 may return (1080, 1920, 3).

- `Height`: number of pixel rows in the image or the number of pixels in each column of the image array.
- `Width`: number of pixel columns in the image or the number of pixels in each row of the image array.
- `Number of channels`: number of components used to represent each pixel.

</details>

<details><summary><strong>cv2.resize(src, dsize, [dst], [fx], [fy], [interpolation])</strong></summary>

<br>

Change the original height and/or width of a source image.

This method accepts two required and four optional parameters:

- `src`: input image or frame.
- `dsize`: desired height and width of the output image in the form of a tuple.
- `dst` (Optional): destination output image.
- `fx` (Optional): scale factor along the horizontal axis.
- `fy` (Optional): scale factor along the vertical axis.
- `intepolation` (Optional): behavior of neighboring pixels when increasing or decreasing the size of an image. This flag accepts the following options:
  - `cv2.INTER_NEAREST`: finds the “nearest” neighboring pixel and assumes the intensity value. Often results in relatively poor image quality and “blocky” artifacts.
  - `cv2.INTER_LINEAR` (default): takes neighboring pixels and uses this neighborhood to calculate the interpolated value (rather than just assuming the nearest pixel value).
  - `cv2.INTER_AREA`: resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moiré-free results. But when the image is zoomed, it is similar to the `cv2.INTER_NEAREST` method.
  - `cv2.INTER_CUBIC`: a bicubic interpolation over 4 x 4 pixel neighborhood.
  - `cv2.INTER_LANCSOZ4`: a Lanczos interpolation over 8×8 pixel neighborhood.

[This article shows examples of how the different interpolation methods may affect the quality of the image.](https://chadrick-kwag.net/cv2-resize-interpolation-methods/)

</details>

<br>

## Resizing and Rescaling a Video

```py
video = cv2.VideoCapture('cute_dog.mp4')

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv.INTER_AREA)

while (video.isOpened()):
    isTrue, frame = video.read()
    rescaled_frame = rescaleFrame(frame, 0.5)
    cv2.imshow('Cute Dog', rescaled_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
```

## Resizing and Rescaling a Live Video

```py
video = cv2.VideoCapture(0)

while (video.isOpened()):
    isTrue, frame = video.read()
    resized_frame = imutils.resize(frame, width=320)
    cv2.imshow('Normal', frame)
    cv2.imshow('Resized', resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
```

<details><summary><strong>video.isOpened()</strong></summary>

<br>

Returns `True` if video capturing has been initialized.

</details>

<details><summary><strong>imutils.resize(src, [width], [height], [inter])</strong></summary>

<br>

`imutils.resize` function maintains the aspect ratio and provides the keyword arguments `width` and `height` so the image can be resized to the intended width/height while (1) maintaining aspect ratio and (2) ensuring the dimensions of the image do not have to be explicitly computed by the developer.

This method accepts one required and three optional parameters:

- `src`: input image or frame.
- `width` (Optional): desired width of the resulting ouput.
- `height` (Optional): desired height of the resulting ouput.
- `inter` (Optional): interpolation. Behavior of neighboring pixels when increasing or decreasing the size of an image. This flag accepts the following options:
  - `cv2.INTER_NEAREST`: finds the “nearest” neighboring pixel and assumes the intensity value. Often results in relatively poor image quality and “blocky” artifacts.
  - `cv2.INTER_LINEAR` (default): takes neighboring pixels and uses this neighborhood to calculate the interpolated value (rather than just assuming the nearest pixel value).
  - `cv2.INTER_AREA`: resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moiré-free results. But when the image is zoomed, it is similar to the `cv2.INTER_NEAREST` method.
  - `cv2.INTER_CUBIC`: a bicubic interpolation over 4 x 4 pixel neighborhood.
  - `cv2.INTER_LANCSOZ4`: a Lanczos interpolation over 8×8 pixel neighborhood.

[This article shows examples of how the different interpolation methods may affect the quality of the image.](https://chadrick-kwag.net/cv2-resize-interpolation-methods/)

</details>

<br>

## Cropping an Image

```py
img = cv2.imread('cute_cat.jpeg')
cropped = img[800:1500, 900:1600]
cv2.imshow('Cute Cat', img)
cv2.imshow('Cropped', cropped)
cv2.waitKey(0)
```

## Drawing Shapes on Images

### Creating a Blank Image

```py
blank = numpy.zeros((500, 500, 3), dtype='uint8')
```

<details><summary><strong>numpy.zeros(shape, [dtype], [order])</strong></summary>

<br>

Returns new array of given shape and type, filled with zeros.

This method accepts one required and two optional parameters:

- `shape`: integer or sequence of integers.
  ```py
  array_1d = numpy.zeros(3)
  # Returns
  [0. 0. 0.]
  ```
  ```py
  array_2d = numpy.zeros((2, 3), dtype=int)
  # Returns
  [[0 0 0]
  [0 0 0]]
  ```
  ```py
  array_mix_type = np.zeros((2, 2), dtype=[('x', 'int'), ('y', 'float')])
  # Returns
  [[(0, 0.) (0, 0.)]
  [(0, 0.) (0, 0.)]]
  ```
- `dtype` (Optional): desired data-type for the returned array. The default value is `float64`.
- `order` (Optional): whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory.

</details>

<br>

### Drawing a Line

```py
cv2.line(blank, (100, 250), (300, 400), (255, 255, 255), thickness=3)
cv2.imshow('Line', blank)
```

<details><summary><strong>cv2.line(image, start_point, end_point, color, thickness)</strong></summary>

<br>

Draw a straight line on an image.

This method accepts five required parameters:

- `image`: source image on which the line will be drawn.
- `start_point`: starting point coordinate of the line. Coordinates are represented as a tuple of pixels, i.e. (x_coordinate, y_coordinate).
- `end_point`: ending point coordinate of the line. Coordinates are represented as a tuple of pixels, i.e. (x_coordinate, y_coordinate).
- `color`: color of the line to be drawn. BGR format, tuple.
- `thickness`: thickness of the line in pixels.

</details>

<br>

### Drawing a Rectangle

```py
cv2.rectangle(blank, (0, 0), (499, 250), (0, 255, 0), thickness=1)
cv2.imshow('Rectangle', blank)
```

<details><summary><strong>cv2.rectangle(image, start_point, end_point, color, thickness)</strong></summary>

<br>

Draw a rectangle on an image.

This method accepts five required parameters:

- `image`: source image on which the rectangle will be drawn.
- `start_point`: starting point coordinate of the rectangle. Coordinates are represented as a tuple of pixels, i.e. (x_coordinate, y_coordinate).
- `end_point`: ending point coordinate of the rectangle. Coordinates are represented as a tuple of pixels, i.e. (x_coordinate, y_coordinate).
- `color`: color of the rectangle border line to be drawn. BGR format, tuple.
- `thickness`: thickness of the rectangle border line in pixels. Thickness of -1 px will fill the rectangle shape with the specified color.

</details>

<br>

### Drawing a Circle

```py
cv2..circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0, 0, 255), thickness=3)
cv2.imshow('Circle', blank)
```

<details><summary><strong>cv2.circle(image, center_coordinates, radius, color, thickness)</strong></summary>

<br>

Draw a circle on an image.

This method accepts five required parameters:

- `image`: source image on which the rectangle will be drawn.
- `center_ccordinates`: center coordinate of the rectangle. Coordinates are represented as a tuple of pixels, i.e. (x_coordinate, y_coordinate).
- `radius`: radius of the circle in pixels.
- `color`: color of the circle border line to be drawn. BGR format, tuple.
- `thickness`: thickness of the circle border line in pixels. Thickness of -1 px will fill the rectangle shape with the specified color.

</details>

<br>

## Adding Text to Images

```py
cv2.putText(blank, 'Hello World', (150, 255), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), thickness=2)
cv2.imshow('Text', blank)
cv2.waitKey(0)
```

<details><summary><strong>cv2.putText(image, text, org, font, fontScale, color, thickness, [lineType], [bottomLeftOrigin])</strong></summary>

<br>

Draw a text string on an image.

This method accepts seven required and two optional parameters:

- `image`: source image on which the rectangle will be drawn.
- `text`: text string to be drawn on the image.
- `org`: the coordinates of bottom-left corner of text. Coordinates are represented as a tuple of pixels, i.e. (x_coordinate, y_coordinate).
- `font`: denotes the font type. Example: cv2.FONT_HERSHEY_TRIPLEX.
- `fontScale`: font scale factor that is multiplied by the font-specific base size.
- `color`: color of the text to be drawn. BGR format, tuple.
- `thickness`: thickness of the text in pixels.
- `lineType` (Optional): type of the line to be used to draw text. Possible values:
  - `cv2.FILLED`
  - `cv2.LINE_4`
  - `cv2.LINE_8`
  - `cv2.LINE_AA`
- `bottomLeftOrigin` (Optional): when true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner. True by default.

</details>

<br>

## Converting an Image to Grayscale

```py
img = cv2.imread('cute_cat.jpeg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Cute Cat Grayscale', img_gray)
cv2.waitKey(0)
```

<details><summary><strong>cv2.cvtColor(src, code, [dst], [dstCn])</strong></summary>

<br>

Convert an image from one color space to another. There are more than 150 color-space conversion methods available in OpenCV.

This method accepts two required and two optional parameters:

- `src`: source image which will be converted to grayscale.
- `code`: color space conversion code.
- `dst` (Optional): output image of the same size and depth as source image.
- `dstCn` (Optional): number of channels in the destination image. If the parameter is 0 then the number of the channels is derived automatically from source and code.

</details>

<br>

## Blurring an Image

```py
img = cv2.imread('cute_cat.jpeg')
blurred_cat = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
cv2.imshow('Blurred Cat', blurred_cat)
cv2.waitKey(0)
```

<details><summary><strong>cv2.GaussianBlur(src, ksize, [dst], [sigmaX], [sigmaY], [borderType])</strong></summary>

<br>

Apply Gaussian blur on input image. The blurring of an image means smoothening of an image i.e., removing outlier pixels that may be noise in the image.

This method accepts two required and four optional parameters:

- `src`: source image to which blur will be applied.
- `ksize`: kernel size. Kernal is matrix of an (no. of rows)\*(no. of columns) order. Its size is given in the form of tuple (no. of rows, no. of columns). no. of rows and no. of columns should be odd. If ksize is set to (0 0), then ksize is computed from sigma values.
- `dst` (Optional): output image.
- `sigmaX` (Optional): standard deviation value of kernal along horizontal direction.
- `sigmaY` (Optional): standard deviation value of kernal along vertical direction.
- `borderType` (Optional): specifies image boundaries while kernel is applied on image borders. Possible values:
  - `cv2.BORDER_CONSTANT`
  - `cv2.BORDER_REPLICATE`
  - `cv2.BORDER_REFLECT`
  - `cv2.BORDER_WRAP`
  - `cv2.BORDER_REFLECT_101`
  - `cv2.BORDER_TRANSPARENT`
  - `cv2.BORDER_REFLECT101`
  - `cv2.BORDER_DEFAULT`
  - `cv2.BORDER_ISOLATED`

</details>

<br>

## Edge Cascade, Image Dilation and Erosion

```py
img = cv2.imread('cute_cat.jpeg')
# Note: passing a blurred image decreases the amount of detected edges.
canny = cv2.Canny(img, 125, 175)
cv2.imshow('Edges', canny)
# Dilating
dilated = cv2.dilate(canny, (3, 3), iterations=1)
cv2.imshow('Dilated', dilated)
# Eroding
eroded = cv2.erode(dilated, (3, 3), iterations=1)
cv2.imshow('Eroded', eroded)
cv2.waitKey(0)
```

<details><summary><strong>cv2.Canny(image, threshold1, threshold2, [apertureSize], [L2gradient])</strong></summary>

<br>

This method uses canny edge detection algorithm for finding the edges in the image.

This method accepts three required and two optional parameters:

- `image`: source image to be used for edge detection.
- `threshold1`: the High threshold value of intensity gradient.
- `threshold2`: the Low threshold value of intensity gradient.
- `apertureSize` (Optional): order of Kernel (matrix) for the Sobel filter. Default value is (3 x 3). Value should be odd between 3 and 7. Used for finding image gradients. Filter is used for smoothening and sharpening of an image.
- `L2gradient` (Optional): specifies the equation for finding gradient magnitude. L2gradient is of boolean type, and its default value is False.

</details>

<details><summary><strong>cv2.dilate(image, kernel , [dst], [anchor], [iterations], [borderType], [borderValue])</strong></summary>

<br>

This method is used to increase object area and accentuate features. A pixel element in the original image is ‘1’ if at least one pixel under the kernel is ‘1’. In cases like noise removal, erosion is followed by dilation. Since erosion removes white noises, it also shrinks the object, so we dilate it. Since noise is gone we can increase our object area without the noise coming back.

This method accepts two required and five optional parameters:

- `image`: source image to be dilated.
- `kernel`: the matrix of odd size (3,5,7) to be convolved with the image.
- `dst` (Optional): output image.
- `anchor` (Optional): variable of type integer representing the anchor point. Default value is (-1, -1) meaning the anchor is at the kernel center.
- `iterations` (Optional): integer value which determine how much you want to dilate a given image.
- `borderType` (Optional): depicts the kind of border to be added. Possible values:
  - `cv2.BORDER_CONSTANT`
  - `cv2.BORDER_REPLICATE`
  - `cv2.BORDER_REFLECT`
  - `cv2.BORDER_WRAP`
  - `cv2.BORDER_REFLECT_101`
  - `cv2.BORDER_TRANSPARENT`
  - `cv2.BORDER_REFLECT101`
  - `cv2.BORDER_DEFAULT`
  - `cv2.BORDER_ISOLATED`
- `borderValue` (Optional): border value in case of a constant border.

</details>

<details><summary><strong>cv2.erode(image, kernel , [dst], [anchor], [iterations], [borderType], [borderValue])</strong></summary>

<br>

This method is used to erode away the boundaries of foreground objects. A pixel in the original image (either 1 or 0) will be considered 1 only if all the pixels under the kernel are 1, otherwise, it is eroded (made to zero). Thus all the pixels near the boundary will be discarded depending upon the size of the kernel. Useful for removing small white noises. Also used to detach connected objects.

This method accepts two required and five optional parameters:

- `image`: source image to be eroded.
- `kernel`: the matrix of odd size (3,5,7) to be convolved with the image.
- `dst` (Optional): output image.
- `anchor` (Optional): variable of type integer representing the anchor point. Default value is (-1, -1) meaning the anchor is at the kernel center.
- `iterations` (Optional): integer value which determine how much you want to dilate a given image.
- `borderType` (Optional): depicts the kind of border to be added. Possible values:
  - `cv2.BORDER_CONSTANT`
  - `cv2.BORDER_REPLICATE`
  - `cv2.BORDER_REFLECT`
  - `cv2.BORDER_WRAP`
  - `cv2.BORDER_REFLECT_101`
  - `cv2.BORDER_TRANSPARENT`
  - `cv2.BORDER_REFLECT101`
  - `cv2.BORDER_DEFAULT`
  - `cv2.BORDER_ISOLATED`
- `borderValue` (Optional): border value in case of a constant border.

</details>

<br>
