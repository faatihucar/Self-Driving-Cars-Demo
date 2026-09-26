import cv2
import imutils
import numpy as np

path="test_videos/road.mp4"

cap = cv2.VideoCapture(path)

ret, frame = cap.read()

top_left=(252,292)
bottom_left=(158,354)

top_right=(361,292)
bottom_right=(440,354)

pts1=np.float32([top_left,bottom_left,top_right,bottom_right])
pts2=np.float32([[0,0],[0,360],[640,0],[640,360]])
color=(0,0,255)

#sliding window parameter


def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H","Trackbars",0,255,nothing)
cv2.createTrackbar("L-S","Trackbars",0,255,nothing)
cv2.createTrackbar("L-V","Trackbars",200,255,nothing)

cv2.createTrackbar("U-H","Trackbars",255,255,nothing)
cv2.createTrackbar("U-S","Trackbars",50,255,nothing)
cv2.createTrackbar("U-V","Trackbars",255,255,nothing)


while True:
    ret, frame = cap.read() 
    frame = imutils.resize(frame,width=640)
    frame_copy=frame.copy()

    cv2.circle(frame,top_left,5,color,-1)
    cv2.circle(frame,bottom_left,5,color,-1)
    cv2.circle(frame,top_right,5,color,-1)
    cv2.circle(frame,bottom_right,5,color,-1)

    #Bird's Eye View için gerekli dönüşüm matrisi
    #pts1 = ROI       pts2=Görüntüyü tanıyan noktalar

    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    transformed_frame = cv2.warpPerspective(frame_copy,matrix,(640,360))

    transformed_frame_hsv=cv2.cvtColor(transformed_frame,cv2.COLOR_BGR2HSV)

    LOWER_H = cv2.getTrackbarPos("L-H","Trackbars")
    LOWER_S = cv2.getTrackbarPos("L-S","Trackbars")
    LOWER_V = cv2.getTrackbarPos("L-V","Trackbars")

    UPPER_H = cv2.getTrackbarPos("U-H","Trackbars")
    UPPER_S = cv2.getTrackbarPos("U-S","Trackbars")
    UPPER_V = cv2.getTrackbarPos("U-V","Trackbars")

    LOWER = np.array([LOWER_H,LOWER_S,LOWER_V])
    UPPER= np.array([UPPER_H,UPPER_S,UPPER_V])

    transformed_frame_mask = cv2.inRange(transformed_frame_hsv,LOWER,UPPER)

    histogram=np.sum(transformed_frame_mask[transformed_frame_mask.shape[0]//2:, :],axis=0)
    middle_point=np.int32(histogram.shape[0]/2)

    left_side=np.argmax(histogram[:middle_point])
    right_side=np.argmax(histogram[middle_point:]) + middle_point

    left_x=[]
    right_x=[]

    transformed_frame_mask_copy=transformed_frame_mask.copy()

    starting_y=360

    while starting_y>0:
        #left side
        img = transformed_frame_mask[starting_y-40:starting_y , left_side-50:left_side+50]
        contours,_=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                center_x,center_y=np.int32(M["m10"]/M["m00"]),np.int32(M["m01"]/M["m00"])
                left_side=left_side-50+center_x
        
        #right side
        img = transformed_frame_mask[starting_y-40:starting_y , right_side-50:right_side+50]
        contours,_=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                center_x,center_y=np.int32(M["m10"]/M["m00"]),np.int32(M["m01"]/M["m00"])
                right_side=right_side-50+center_x
        

        cv2.rectangle(transformed_frame_mask_copy,(left_side-60,starting_y),(left_side+60,starting_y-40),(255,255,255),2)
        cv2.rectangle(transformed_frame_mask_copy,(right_side-60,starting_y),(right_side+60,starting_y-40),(255,255,255),2)

        starting_y = starting_y-40


    #cv2.imshow("Test",frame)
    cv2.imshow("Bird's Eye View - BGR",transformed_frame)
    cv2.imshow("Bird's Eye View - HSV",transformed_frame_hsv)
    cv2.imshow("Bird's Eye View - MASK",transformed_frame_mask)
    cv2.imshow("Sliding Window",transformed_frame_mask_copy)
    if cv2.waitKey(10)==27:
        break

cv2.destroyAllWindows()