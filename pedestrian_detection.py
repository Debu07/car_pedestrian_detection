import cv2
import time

full_body_classifier=cv2.CascadeClassifier('haarcascade_fullbody.xml')
cap=cv2.VideoCapture('crossing.mp4')

while cap.isOpened():
    time.sleep(0.5)
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cars=full_body_classifier.detectMultiScale(gray,1.4,2)

    for x,y,w,h in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
        cv2.imshow("pedestrians detected:",frame)
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()