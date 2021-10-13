import cv2 as cv

# Reading and displaying an image.
img = cv.imread('cute_cat.jpeg')
cv.imshow('Cute Cat', img)
cv.waitKey(0)

# Resizing and rescaling images.
img = cv.imread('cute_cat.jpeg')


def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


cv.imshow('Cute Cat', rescaleFrame(img))
cv.waitKey(0)
