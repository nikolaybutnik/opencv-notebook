import cv2 as cv
import numpy as np

# Drawing on images
# Create a blank image
blank = np.zeros((500, 500, 3), dtype='uint8')
# numpy.zeros() returns a new array of given shape and type, with zeros.
cv.imshow('Blank', blank)

# Paint the image a certain color
# [:] is the array slice syntax for every element in the array.
blank[:] = 0, 255, 0
cv.imshow('Green', blank)

# Paint some of the pixels in an image
blank[200:300, 300:400] = 0, 0, 255
cv.imshow('Red', blank)

# Draw a rectangle
# Specyfy start/end coordinates, color, and line thickness
# cv.rectangle(blank, (0, 0), (499, 250), (0, 255, 0), thickness=1)
# To fill the rectangle use thickness=cv.FILLED or thickness=-1
# Alternatively, you can select pixels relative to the shape
cv.rectangle(
    blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (0, 255, 0), thickness=cv.FILLED)
cv.imshow('Rectangle', blank)

# Draw a circle
cv.circle(blank, (blank.shape[1]//2,
          blank.shape[0]//2), 40, (0, 0, 255), thickness=3)
cv.imshow('Circle', blank)

# Draw a line
cv.line(blank, (100, 250), (300, 400), (255, 255, 255), thickness=3)
cv.imshow('Line', blank)

# Write text on an image
cv.putText(blank, 'Hello World', (150, 255),
           cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), thickness=2)
cv.imshow('Text', blank)

cv.waitKey(0)
