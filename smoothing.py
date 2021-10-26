import cv2 as cv


img = cv.imread('cute_cat.jpeg')
cv.imshow('Photo', img)

# Average Blur
img_average = cv.blur(img, (3, 3))
# cv.imshow('Average Blur', img_average)

# Gaussian Blur
img_gaussian = cv.GaussianBlur(img, (3, 3), 0)
# cv.imshow('Gaussian Blur', img_gaussian)

# Median Blur
img_median = cv.medianBlur(img, 3)
cv.imshow('Median Blur', img_median)

# Bilateral Blur
img_bilateral = cv.bilateralFilter(img, 5, 15, 15)
cv.imshow('Bilaeral Blur', img_bilateral)

cv.waitKey(0)
