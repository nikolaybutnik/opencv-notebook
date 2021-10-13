import cv2 as cv

# Reading and displaying a video.
video = cv.VideoCapture('cute_dog.mp4')
while True:
    isTrue, frame = video.read()
    cv.imshow('Cute Dog', frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
video.release()
cv.destroyAllWindows()
