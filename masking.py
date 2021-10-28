import cv2 as cv
import numpy as np

img = cv.imread('cute_cat.jpeg')
blank = np.zeros(img.shape[:2], dtype='uint8')

mask = cv.circle(
    blank, (img.shape[1]//2 + 75, img.shape[0]//2 + 320), 400, 255, -1)

masked_img = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Masked Image', masked_img)

cv.waitKey(0)
