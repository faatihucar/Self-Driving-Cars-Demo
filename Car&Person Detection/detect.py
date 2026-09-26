#library
import cv2
import numpy as np
import random
import time
from ultralytics import YOLO


#predefined variable
color=(0,0,255)
font=cv2.FONT_HERSHEY_SIMPLEX
confidence_score=0.5
text_color_b=(0,0,0)
text_color_w=(255,255,255)
background_color = (0,255,0)
class_ids = [0, 1, 2 , 3 , 5 , 6 , 7 ,8]
total_fps=0
average_fps=0
number_of_frame=0
video_frames=[]
save_path="results/test_vid_res.avi"


#load model
model=YOLO("models/yolo11l.pt")
labels=model.names
colors=[[random.randint(0,255) for _ in range(0,3)] for _ in labels]

#print(f"number of class {len(labels)} ")
#print(f"number of colors {len(colors)} ")

#load video
video_path="inference/test.mp4"
cap=cv2.VideoCapture(video_path)

video_width = int(cap.get(3)) #video en
video_height = int(cap.get(4)) #video yükseklik
total_video_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #videonun toplam frame sayısı

print(f"video_width: {video_width} ")
print(f"video_height: {video_height} ")
print(f"total_video_frame: {total_video_frame} ")

while True:
    start = time.time()
 
    ret,frame=cap.read()
    
    if ret==False:
        break
    

    results = model(frame,verbose=False)[0]
    
    #bbox, score, class_id

    #boxes değişkeni koordinatları,confidence ve class id'leri içerir
    boxes=np.array(results.boxes.data.tolist()) #işlemek için numpy arrayına çevirmek gerek.

    #birden fazla box olabilir birden fazla tespit etme yapılabilir.

    for box in boxes:
       #print(f"box: {box}")
        x1, y1, x2, y2, score, class_id = box
        x1, y1, x2, y2,class_id = int(x1), int(y1), int(x2), int(y2),int(class_id)

        box_color=colors[class_id]

        if score>confidence_score and class_id in class_ids:
            cv2.rectangle(frame,(x1,y1),(x2,y2),box_color,2)
            
            score=score*100

            class_name=results.names[class_id]

            #text yerleştirme
            text=f"{class_name}: % {score:.2f}"
            text_loc= x1, y1-10 
            labelSize,baseLine=cv2.getTextSize(text,font,1,1)
            cv2.rectangle(frame,
                          (x1, y1-10-labelSize[1]),
                          (x1 + labelSize[0], int(y1+baseLine-10)),
                            box_color,
                            cv2.FILLED)
            
            cv2.putText(frame,text,(x1,y1-10),font,1,text_color_w,thickness=1)
           

    end = time.time()
    number_of_frame += 1
    fps=1/(end-start)
    total_fps=total_fps+fps
    average_fps=total_fps/number_of_frame
    average_fps=float("{:.2f}".format(average_fps))
    cv2.rectangle(frame,(10,2),(280,50),background_color,-1)
    cv2.putText(frame,"FPS: "+str(average_fps),(20,40),font,1.5,text_color_b,thickness=3)
        

    video_frames.append(frame)
    print("(%2d / %2d) Frame Processed" % (number_of_frame,total_video_frame))
    cv2.imshow("Test",frame)

    if cv2.waitKey(20) & 0xFF==ord("q"):
        break

cap.release()

cv2.destroyAllWindows()


print("Video is creating please wait ! ")

#Video writer

video_writer = cv2.VideoWriter(save_path,
                               cv2.VideoWriter_fourcc(*'XVID'),
                               int(average_fps),
                               (video_width,video_height))

for frame in video_frames:
    video_writer.write(frame)

video_writer.release()

print(f"Video is saved to {save_path}")


################
# # imgsz
# # Zero Copy
# # Video Size 
# # YOLOv8 Arch.
# # Hyperparameter 
# # TensorRT, OpenVINO
# # Combination

# # Performance Analysis
################