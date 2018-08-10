import cv2
import numpy as np

vid = cv2.VideoCapture(2)


while True:
    ret, frame = vid.read()
    cv2.imshow('Display', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
