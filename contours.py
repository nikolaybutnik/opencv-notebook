import cv2 as cv
import numpy as np

# Contour Detection - Canny
img = cv.imread('photography.jpeg')
img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_blurred = cv.GaussianBlur(img_grayscale, (3, 3), cv.BORDER_DEFAULT)
canny = cv.Canny(img_blurred, 125, 175)

contours, hierarchy = cv.findContours(
    canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found')

cv.imshow('Canny', canny)

cv.waitKey(0)

# Contour Detection - Threshold
img = cv.imread('photography.jpeg')
img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blank_img = np.zeros(img.shape, dtype='uint8')

ret, thresh = cv.threshold(img_grayscale, 125, 255, cv.THRESH_BINARY)

contours, hierarchy = cv.findContours(
    thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found')

cv.drawContours(blank_img, contours, -1, (0, 0, 255), 1)
cv.imshow('Contours Drawn', blank_img)

cv.waitKey(0)
