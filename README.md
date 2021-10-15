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

- `Path`: the path to the specified image in a string format.
- `Flag`: specifies the way in which the image should be read. The default flag is `cv2.IMREAD_COLOR`.

These are the three possible flag parameters for the method:

- `cv2.IMREAD_COLOR` (default): It specifies to load a color image. Any transparency of image will be neglected. Alternatively, we can pass integer value 1 for this flag.
- `cv2.IMREAD_GRAYSCALE`: It specifies to load an image in grayscale mode. Alternatively, we can pass integer value 0 for this flag.
- `cv2.IMREAD_UNCHANGED`: It specifies to load an image as such including alpha channel. Alternatively, we can pass integer value -1 for this flag.

</details>

<details><summary><strong>cv2.imshow(window_name, source)</strong></summary>

<br>

Display an image in a new window. The window will automatically scale to the image size.

This method accepts two parameters:

- `Window name`: a string respresenting the name of the window in which the image will be displayed.
- `Source`: the image that will be displayed in the window.

</details>

<details><summary><strong>cv2.waitKey(delay)</strong></summary>

<br>

Necessary to avoid the script from immediately terminating.

The method accepts a delay input in milliseconds. This is the time that the script will wait for the program to continue. If `0` is passed, the program will wait for input indefinitely. In this case, if waitkey is not used, the program will automatically terminate after the imshow line runs, resulting in the image flashing in screen for a fraction of a second. Passing a `0` wil ensure the image stays on screen until the user chooses to close it.

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

<details><summary><strong>cv2.resize(source, desired_size, [fx], [fy], [interpolation])</strong></summary>

<br>

Change the original height and/or width of a source image.

This method accepts two required and three optional parameters:

- `Source`: input image or frame.
- `Desired size`: Desired height and width of the output image in the form of a tuple.
- `Fx` (Optional): scale factor along the horizontal axis.
- `Fy` (Optional): scale factor along the vertical axis.
- `Intepolation` (Optional): Behavior of neighboring pixels when increasing or decreasing the size of an image. This flag accepts the following options:
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

<details><summary><strong>imutils.resize(source, [width], [height], [inter])</strong></summary>

<br>

`imutils.resize` function maintains the aspect ratio and provides the keyword arguments `width` and `height` so the image can be resized to the intended width/height while (1) maintaining aspect ratio and (2) ensuring the dimensions of the image do not have to be explicitly computed by the developer.

This method accepts one required and three optional parameters:

- `Source`: input image or frame.
- `Width` (Optional): desired width of the resulting ouput.
- `Height` (Optional): desired height of the resulting ouput.
- `Intepolation` (Optional): Behavior of neighboring pixels when increasing or decreasing the size of an image. This flag accepts the following options:
  - `cv2.INTER_NEAREST`: finds the “nearest” neighboring pixel and assumes the intensity value. Often results in relatively poor image quality and “blocky” artifacts.
  - `cv2.INTER_LINEAR` (default): takes neighboring pixels and uses this neighborhood to calculate the interpolated value (rather than just assuming the nearest pixel value).
  - `cv2.INTER_AREA`: resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moiré-free results. But when the image is zoomed, it is similar to the `cv2.INTER_NEAREST` method.
  - `cv2.INTER_CUBIC`: a bicubic interpolation over 4 x 4 pixel neighborhood.
  - `cv2.INTER_LANCSOZ4`: a Lanczos interpolation over 8×8 pixel neighborhood.

[This article shows examples of how the different interpolation methods may affect the quality of the image.](https://chadrick-kwag.net/cv2-resize-interpolation-methods/)

</details>
