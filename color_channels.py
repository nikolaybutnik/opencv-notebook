import cv2 as cv
import numpy as np

img = cv.imread('photography.jpeg')

b, g, r = cv.split(img)
# The colors are represented in grayscale with intensities. Example: if there's a lot of red in the r image, it'll be shown as white.
# Other colors will show as varying degrees of gray/black.
cv.imshow('Blue', b)
cv.imshow('Green', g)
cv.imshow('Red', r)

# Note: if we print the shape og img, we'll see that the third element in the tuple will be 3. That is the color channel.
# Printing b, g, and r will result in third element in the tuple ot be missing, meaning it's a 1. Therefore, 1 color channel.

img_merged = cv.merge([b, g, r])
cv.imshow('Merged Photo', img_merged)

# There is a way to show the color channels separately. They need to be overlayed on a blank matrix.
blank = np.zeros((img.shape[:2]), dtype='uint8')
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])
cv.imshow('Blue Only', blue)
cv.imshow('Green Only', green)
cv.imshow('Red Only', red)

cv.waitKey(0)
