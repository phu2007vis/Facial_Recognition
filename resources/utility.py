# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午2:13
# @Author : zhuying
# @Company : Minivision
# @File : utility.py
# @Software : PyCharm
import base64
from datetime import datetime
import os
import cv2
import numpy as np
import yaml
import img64
from datetime import datetime
import torch 
from sklearn.metrics import accuracy_score
SPLIT = "()()()()()()()"
name_real_or_fake = ['fake','real','fake']

def check_box(box):
    x1,y1,x2,y2 = box
    if (x2-x1)*(y2-y1) < 50:
        return False
    return True

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
def encode_image(image):
    #buffer
    return cv2.imencode(".png",image)[1]
def image_to_base64(image):
    '''
    image bgr
    rgb base 64
    '''
    return img64.image_to_base64(image)
def base64_to_image(base64_string):
    # Remove the data URL prefix if it exists
    if base64_string.startswith("data:image"):
        base64_string = base64_string.split(",")[1]
    # Decode the base64 string
    decoded_data = base64.b64decode(base64_string)
    # Convert to numpy array
    nparr = np.frombuffer(decoded_data, np.uint8)
    # Decode the image using cv2
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    '''
    bgr image
    '''
    return image

def get_time():
    #YYYY-MM-DD format
    current_datetime = datetime.now()
    return current_datetime.strftime('%Y-%m-%d')

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def check_check_in(old_attendence,new_atendence,distance = 20):
    id,_,old_time = old_attendence
    new_id,_,new_time = new_atendence
    old_time,new_time = float(old_time),float(new_time)
    if id != new_id:
        return True
    if abs(new_time-old_time) >=distance:
        return True
    return False
def valid_accuracy(net,ds,y = None):
    y = [int(y.clone().detach().max(0)[1].item()) for _,y in ds]
    y_predict = []
    with torch.no_grad():
        for predict in net.forward_iter(ds):
            y_predict.extend(predict.max(1)[1].tolist())
    return accuracy_score(y,y_predict)

def rename_folder(folder_path):
    import glob
    for i,path_dir in enumerate(glob.glob(folder_path)):
        try:
            int(os.base_name(path_dir.spit("-->")[0]))
        except:
            base_name = str(i) +SPLIT+os.path.basename(path_dir).split(SPLIT)[-1]
            dir_name = os.path.dirname(path_dir)
            new_name = os.path.join(dir_name,base_name)
            os.rename(path_dir,new_name)


def cosine_similarity(vector1, vector2):
    """
    Compute cosine similarity between two vectors.

    Parameters:
    vector1 (numpy.ndarray): First vector.
    vector2 (numpy.ndarray): Second vector.

    Returns:
    float: Cosine similarity between the two vectors.
    """
    vector1 = vector1.squeeze()
    vector2 = vector2.squeeze()
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    return dot_product / (norm_vector1 * norm_vector2)

   

    