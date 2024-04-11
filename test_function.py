from resources.face_image.extract_face import extrace_face,get_dlib_full_object_detections,extract_feature,extract_feature_pipeline
import cv2
import numpy as np

im1 = cv2.imread(r"D:\face_liveness_detection-Anti-spoofing\DataBase\0()()()()()()()phuoc\WIN_20240322_19_24_36_Pro.jpg")
im2 = cv2.imread(r"D:\face_liveness_detection-Anti-spoofing\DataBase\2()()()()()()()thanh\WIN_20240111_13_59_46_Pro.jpg")

type  = "face_alignment"
import time
t1 = time.time()
feature1 = extract_feature_pipeline(im1,type=type)
t2 = time.time()
feature2 = extract_feature_pipeline(im2,type=type)
t3  = time.time()
print(t2-t1)
print(t3-t2)
print(np.sqrt(np.sum((feature1-feature2)**2)))
