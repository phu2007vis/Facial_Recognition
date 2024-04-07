from resources.detection_model.detect import face_detect
from resources.utility import *
from deepface  import DeepFace
import numpy as np
model   = DeepFace.build_model("Age")
import cv2
cam = cv2.VideoCapture(0)
while True:
    try:
        ret,frame = cam.read()
        box = face_detect(frame)[0]
        if check_box(box):
            x1,y1,x2,y2 = box
            input = np.expand_dims(cv2.resize(frame[y1:y2,x1:x2,:],(224,224)),axis=0)
            age_predict = model.predict(input)
            frame = put_text(frame,str(age_predict))
        cv2.imshow("frame",frame)
        if cv2.waitKey(8) == ord("q"):
            break   
    except:
        pass
    