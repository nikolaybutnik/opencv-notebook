import cv2 as cv

# Reading and displaying an image.
img = cv.imread('cute_cat.jpeg')
cv.imshow('Cute Cat', img)
cv.waitKey(0)
