import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('photography.jpeg')

# BGR => Grayscale
img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Photo Gray', img_grayscale)

# BGR => HSV
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('Photo HSV', img_hsv)

# BGR => LAB
img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('Photo LAB', img_lab)

# BGR => RGB
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('Photo RGB', img_rgb)
plt.imshow(img_rgb)
plt.show()

cv.waitKey(0)

# Note: not all formats can be converted directly to one another.
# Example: to achieve GRAY => HSV we need to do GRAY => BGR, and then BGR => HSV.
