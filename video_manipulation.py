import cv2 as cv

# Reading and displaying a video.
# video = cv.VideoCapture('cute_dog.mp4')
# while True:
#     isTrue, frame = video.read()
#     cv.imshow('Cute Dog', frame)
#     if cv.waitKey(25) & 0xFF == ord('q'):
#         break
# video.release()
# cv.destroyAllWindows()

# Resizing and rescaling videos.
# video = cv.VideoCapture('cute_dog.mp4')


# def rescaleFrame(frame, scale=0.75):
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)
#     dimensions = (width, height)
#     return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


# while True:
#     isTrue, frame = video.read()
#     rescaled_frame = rescaleFrame(frame, 0.5)
#     cv.imshow('Cute Dog', rescaled_frame)
#     if cv.waitKey(25) & 0xFF == ord('q'):
#         break
# video.release()
# cv.destroyAllWindows()

# Resizng and resaling live camera video.
video = cv.VideoCapture(0)

# This function will only work on live cam video, and not on video files.


def changeRes(width, height):
    video.set(3, width)
    video.set(4, height)


# changeRes(1920, 1080)

while True:
    isTrue, frame = video.read()
    if isTrue:
        cv.imshow('Webcam', frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
video.release()
cv.destroyAllWindows()
