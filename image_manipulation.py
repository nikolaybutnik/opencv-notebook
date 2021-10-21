import cv2 as cv
import numpy as np

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

# Converting image to grayscale
img = cv.imread('cute_cat.jpeg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Cute Cat Grayscale', img_gray)
cv.waitKey(0)

# Blurring an Image
img = cv.imread('cute_cat.jpeg')
blurred_cat = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
cv.imshow('Blurred Cat', blurred_cat)
cv.waitKey(0)

# Edge Cascade
img = cv.imread('cute_cat.jpeg')
# Note: passing a blurred image decreases the amount of detected edges.
canny = cv.Canny(img, 125, 175)
cv.imshow('Edges', canny)
# Dilating
dilated = cv.dilate(canny, (3, 3), iterations=1)
cv.imshow('Dilated', dilated)
# Eroding
eroded = cv.erode(dilated, (3, 3), iterations=1)
cv.imshow('Eroded', eroded)
cv.waitKey(0)

# Cropping
img = cv.imread('cute_cat.jpeg')
cropped = img[800:1500, 900:1600]
cv.imshow('Cute Cat', img)
cv.imshow('Cropped', cropped)
cv.waitKey(0)

# Translating
img = cv.imread('cute_cat.jpeg')


def translate(img, x, y):
    trans_mat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, trans_mat, dimensions)


translated = translate(img, 100, 100)
cv.imshow('Original Cat', img)
cv.imshow('Translated Cat', translated)
cv.waitKey(0)
