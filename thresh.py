import cv2 as cv

img = cv.imread('cute_cat.jpeg')
# cv.imshow('Cute Cat', img)
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Simple thresholding
threshold, thresh = cv.threshold(gray_img, 150, 255, cv.THRESH_BINARY)
cv.imshow('Simple Thershold', thresh)

threshold, thresh_inv = cv.threshold(gray_img, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow('Simple Threshold Inverse', thresh_inv)

# Adaptive thresholding
adaptive_thresh = cv.adaptiveThreshold(
    gray_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
cv.imshow('Adaptive Threshold', adaptive_thresh)

cv.waitKey(0)
