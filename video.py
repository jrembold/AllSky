import cv2
import numpy as np 

vid = cv2.VideoCapture("/home/luke/Dropbox/Thesis/Videos/iridiumflare.avi")
(grabbed,frame) = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
movie = np.expand_dims(frame,0)

x = 0
while grabbed:
    (grabbed,frame) = vid.read()
    if x > 3200:
        movie = np.append(movie,np.expand_dims(frame,0),axis=0)
    # cv2.imshow("pls",frame)
    x += 1
    print(x)
    if cv2.waitKey(2) == ord('q'):
        break
    if x>3550:
        print(movie.shape)
        break

