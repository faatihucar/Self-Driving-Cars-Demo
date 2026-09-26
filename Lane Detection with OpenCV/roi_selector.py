import cv2
import numpy as np
import imutils

path="test_videos/road.mp4"

cap=cv2.VideoCapture(path) 

ret, img = cap.read()
img = imutils.resize(img,width=640)

clicked_points=[]
color=(0,0,255)
font=cv2.FONT_HERSHEY_SIMPLEX

def click_event(event,x,y,flags,param):
    if event== cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),5,color,-1)
        cv2.putText(img,f"{x}, {y}",(x,y-10),font,0.5,color,2)
        cv2.imshow("Test",img)
        clicked_points.append((x,y))


cv2.imshow("Test",img)

cv2.setMouseCallback("Test",click_event)


key = cv2.waitKey(0)
if key == 27:  # ESC tuşu
    cv2.imwrite("coordinates.png", img)
    for point in clicked_points:
        print(f"Coordinate: {point}")

cv2.destroyAllWindows()
