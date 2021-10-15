import cv2 as cv
import imutils

# Reading and displaying a video.
video = cv.VideoCapture('cute_dog.mp4')
while True:
    isTrue, frame = video.read()
    cv.imshow('Cute Dog', frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
video.release()
cv.destroyAllWindows()

# Resizing and rescaling videos.
video = cv.VideoCapture('cute_dog.mp4')


def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


while (video.isOpened()):
    isTrue, frame = video.read()
    resized_frame = rescaleFrame(frame, 0.5)
    cv.imshow('Cute Dog Original', frame)
    cv.imshow('Cute Dog Resized', resized_frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
video.release()
cv.destroyAllWindows()

# Resizng and resaling live camera video.
video = cv.VideoCapture(0)

while (video.isOpened()):
    isTrue, frame = video.read()
    resized_frame = imutils.resize(frame, width=320)
    cv.imshow('Normal', frame)
    cv.imshow('Resized', resized_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv.destroyAllWindows()
