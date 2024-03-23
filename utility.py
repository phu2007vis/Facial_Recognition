# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午2:13
# @Author : zhuying
# @Company : Minivision
# @File : utility.py
# @Software : PyCharm

from datetime import datetime
import os
import cv2
import numpy as np
try:
    import torch
except:
    print("can't import torch")


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


def get_width_height(patch_info):
    w_input = int(patch_info.split('x')[-1])
    h_input = int(patch_info.split('x')[0].split('_')[-1])
    return w_input,h_input


def parse_model_name(model_name):
    info = model_name.split('_')[0:-1]
    h_input, w_input = info[-1].split('x')
    model_type = model_name.split('.pth')[0].split('_')[-1]

    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0])
    return int(h_input), int(w_input), model_type, scale
def xyxy2xywh(box):
    x1,y1,x2,y2 = box
def xywh2xyxy(boxes):
    new_boxes = []
    for box in boxes:
        x,y,w,h = box
        new_box = [x,y,x+w,y+h]
        new_boxes.append(new_box)
    return new_boxes

def make_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def preprocessing_face(face:np.array):
    face = cv2.resize(face,(160,160))
    img_tensor = torch.from_numpy(face.transpose((2, 0, 1))).float()
    img_tensor =  (img_tensor - 127.5) / 128.0
    return img_tensor
def euclit(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)
def put_text(im,text,y = 0,color = (0,0,255)):
    cv2.putText(im,text,(10,50+y),cv2.FONT_HERSHEY_COMPLEX,1,color,2)
    return im
def draw_faces(image,boxes,color =  (0,255,0)):
    for box in boxes:
        x1,y1,x2,y2 = box
        image = cv2.rectangle(image, (x1,y1), (x2,y2), color, 1)
    return image