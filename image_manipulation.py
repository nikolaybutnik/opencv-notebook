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
