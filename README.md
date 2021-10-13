# Getting Started

Install and import the OpenCV library.

```py
pip3 install opencv-python
```

```py
import cv2
```

# Image Manipulation

## Reading and displaying an image

```py
img = cv2.imread('cute_cat.jpeg')
cv2.imshow('Cute Cat', img)
cv2.waitKey(0)
```

<details><summary>cv2.imread(path, flag)</summary>

<br>
Capture an image from a specified file, which you can then assign to a variable. If an image can't be read, this method returns an empty matrix.

This method accepts two parameters:

- Path: the path to the specified image in a string format.
- Flag: specifies the way in which teh image should be read. The default flag is `cv2.IMREAD_COLOR`.

These are the three possible flag parameters for the method:

- `cv2.IMREAD_COLOR` (default): It specifies to load a color image. Any transparency of image will be neglected. Alternatively, we can pass integer value 1 for this flag.
- `cv2.IMREAD_GRAYSCALE`: It specifies to load an image in grayscale mode. Alternatively, we can pass integer value 0 for this flag.
- `cv2.IMREAD_UNCHANGED`: It specifies to load an image as such including alpha channel. Alternatively, we can pass integer value -1 for this flag.

</details>

<details><summary>cv2.imshow(window_name, image)</summary>

<br>
Display an image in a new window. The window will automatically scale to the image size.

This method accepts two parameters:

- Window name: a string respresenting the name of the window in which the image will be displayed.
- Image: the image that will be displayed in the window.

</details>

<details><summary>cv2.waitKey(delay)</summary>

<br>
Necessary to avoid the script from immediately terminating.

The method accepts a delay input in milliseconds. This is the time that the script will wait for the program to continue. If `0` is passed, the program will wait for input indefinitely. In this case, if waitkey is not used, the program will automatically terminate after the imshow line runs, resulting in the image flashing in screen for a fraction of a second. Passing a `0` wil ensure the image stays on screen until the user chooses to close it.
<br><br>

</details>

<br>

# Video Manipulation

## Reading and displaying a video

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

<details><summary>cv2.VideoCapture(path/source)</summary>

<br>
Create a video capture object from a source, which can then be stored in a variable.

This method accepts the source of the video as a parameter. Passing path to a video as a string will allow you to use a local video file. An integer can also be passed, and refers to a camera on the computer. Passing `0` will typically capture video from a webcam. Passing subsequent integers will allow accessing other cameras.
<br><br>

</details>

<details><summary>video.read()</summary>

<br>
Read the video frame by frame. It returns a boolean that tells us whether reading the frame was successful, and the frame itself. The operation needs to be performed inside a while loop.
<br><br>

</details>

<details><summary>cv2.imshow(window_name, image)</summary>

<br>
Displays a frame of the video in a window. The window will automatically scale to the video size.

This method accepts two parameters:

- Window name: a string respresenting the name of the window in which the video will be played.
- Image: the image (frame) that will be displayed in the window.

</details>

<details><summary>video.release()</summary>

<br>
Close video or captruing device. Must be called before creating another instance of the video capture object.
<br><br>

</details>

<details><summary>cv2.destroyAllWindows()</summary>

<br>
Destroys all currently open windows. To destroy a specific window, use the function `cv2.destroyWindow()` where you pass the exact window name.
<br><br>

</details>
