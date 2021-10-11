# Getting Started

Import and install the OpenCV library.

```py
pip3 install opencv-python
```

```py
import cv2
```

# Image Manipulation

## Reading and displaying an image

<details><summary>cv2.imread(path, flag)</summary>
The `cv2.imread()` method captures an image from a specified file, which you can then assign to a variable. If an image can't be read, this method returns an empty matrix.

```py
image = cv2.imread('cute_cat.jpg')
```

This method accepts two parameters:

- Path: the path to the specified image in a string format.
- Flag: specifies the way in which teh image should be read. The default flag is `cv2.IMREAD_COLOR`.

These are the three possible flag parameters for the method:

- `cv2.IMREAD_COLOR` (default): It specifies to load a color image. Any transparency of image will be neglected. Alternatively, we can pass integer value 1 for this flag.
- `cv2.IMREAD_GRAYSCALE`: It specifies to load an image in grayscale mode. Alternatively, we can pass integer value 0 for this flag.
- `cv2.IMREAD_UNCHANGED`: It specifies to load an image as such including alpha channel. Alternatively, we can pass integer value -1 for this flag.

</details>

<details><summary>cv2.imshow(window_name, image)</summary>
The `cv2.imshow()` method displays an image in a new window. The window will automatically scale to the image size.

```py
cv2.imshow('Cute Cat', image)
```

This method accepts two parameters:

- Window name: a string respresenting the name of the window in which the image will be displayed.
- Image: the image that will be displayed in the window.

</details>

<details><summary>cv2.waitKey(delay)</summary>
The `cv2.waitkey()` method is necessary to avoid the script from immediately terminating.

```py
cv2.waikey(0)
```

The method accepts a delay input in milliseconds. This is the time that the script will wait for the program to continue. If `0` is passed, the program will wait for input indefinitely. In this case, if waitkey is not used, the program will automatically terminate after the imshow line runs, resulting in the image flashing in screen for a fraction of a second. Passing a `0` wil ensure the image stays on screen until the user chooses to close it.

</details>
