import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Computing grayscale histograms

img = cv.imread('cute_cat.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blank = np.zeros(img.shape[:2], dtype='uint8')
circle = cv.circle(
    blank, (img.shape[1]//2 + 75, img.shape[0]//2 + 320), 400, 255, -1)
mask = cv.bitwise_and(gray, gray, mask=circle)

gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])
# gray_hist = cv.calcHist([gray], [0], mask, [256], [0, 256])

plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('Number of Pixels')
plt.plot(gray_hist)
plt.xlim([0, 256])
plt.show()

cv.waitKey(0)

# Computing BGR histograms

img = cv.imread('cute_cat.jpeg')

blank = np.zeros(img.shape[:2], dtype='uint8')
mask = cv.circle(
    blank, (img.shape[1]//2 + 75, img.shape[0]//2 + 320), 400, 255, -1)
masked_img = cv.bitwise_and(img, img, mask=mask)

colors = ('b', 'g', 'r')
plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('Number of Pixels')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    # hist = cv.calcHist([img], [i], mask, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.show()

cv.waitKey(0)
