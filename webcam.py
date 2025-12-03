import numpy as np
import cv2


cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret,frame = cap.read()
else:
    ret = False

while ret:
    ret, frame = cap.read()

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()