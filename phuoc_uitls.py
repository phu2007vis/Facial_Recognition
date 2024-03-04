import pickle
import os
import glob
from emotion_detection import f_emotion_detection
import cv2
import numpy as np       
from anti_spoof_predict import model_test,spoof_predict
from utility import xywh2xyxy
emotion_detector         = f_emotion_detection.predict_emotions()
from facenet_pytorch import InceptionResnetV1
import torch
import time
resnet = InceptionResnetV1(pretrained='casia-webface').eval()
def get_face(image,box):
    x1,y1,x2,y2 = box
    return image[y1:y2,x1:x2,:]

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

def preprocessing_face(face:np.array):
    face = cv2.resize(face,(160,160))
    img_tensor = torch.from_numpy(face.transpose((2, 0, 1))).float()
    img_tensor = fixed_image_standardization(img_tensor)
    return img_tensor

def draw_faces(image,boxes,color =  (0,255,0)):
    for box in boxes:
        x1,y1,x2,y2 = box
        image = cv2.rectangle(image, (x1,y1), (x2,y2), color, 1)
    return image
def face_encode(image,box,mode = "rgb"):
    if mode == "bgr":
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #image rgb mode
    x1,y1,x2,y2 = box
    face = image[y1:y2,x1:x2,:]
    face_tensor = preprocessing_face(face)
    with torch.no_grad():
        img_embedding = resnet(face_tensor.unsqueeze(0))
    return np.array(img_embedding[0])
  
def put_text(im,text,y = 0,color = (0,0,255)):
    cv2.putText(im,text,(10,50+y),cv2.FONT_HERSHEY_COMPLEX,1,color,2)
    return im

name_real_or_fake = ['fake','real','fake']
def detect_liveness(im,face_vetification = None,visulize = False):
    boxes_face = xywh2xyxy(model_test[0].get_bbox(im))
    names = []
    emotion = []
    label = []
    if len(boxes_face)!=0:
        boxes_face = [list(boxes_face[0])]
        label,value = spoof_predict(im,boxes_face)
        label = [name_real_or_fake[label]]
        if face_vetification:
            names,_ =  face_vetification.check_face(im,boxes_face,draw_image = True)
        if visulize:
            color = (255,0,0) if label[0] == 'real' else (0,0,255)
            im = draw_faces(im,boxes_face,color)
            im = put_text(im,label[0],color=color)
            im = put_text(im,names[0],50)
    
    # _,emotion = emotion_detector.get_emotion(im,boxes_face)
       
    # {'emotion': ['neutral'], 'boxes_faces': [[143, 150, 350, 358]], 'label': ['real'], 'names': []}
    output = {
        'emotion': emotion,
        'boxes_faces': boxes_face,
        'label': label,
        'names': names
    }
    return im,output

def read_yaml(yaml_path):
    import yaml
    with open(yaml_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data

class FacialVertification:
    def __init__(self,encode_data_path = r"D:\face_liveness_detection-Anti-spoofing\data.pkl",path_data_base  = r"D:\face_liveness_detection-Anti-spoofing\DataBase"):

        self.encode_data_path = encode_data_path
        self.path_data_base = path_data_base
        
        if not os.path.exists(encode_data_path):
            self.gen_data_encode()
        self.names,self.encodes = self._load_data_face(encode_data_path)
    def _load_data_face(self,face_data_path ):

        with open(face_data_path, "rb") as f:
            names = pickle.load(f)
            encodes = pickle.load(f)
        return names,encodes
    def register_new_people(self,name:str,encodings):
        if name in self.names:
            return "Name is already exists",False
        encodings = np.array(encodings)
        assert(len(encodings.shape) == 2)
        if len(encodings) > 0 :
            self.names.extend([name]*len(encodings))
            self.encodes = np.vstack((self.encodes,encodings))
            self.save_data_pkl()
            return "Success registry",True
        return "Camera error",False
        
    def remove_face(self,name_remove):
        idx = [i for i ,name in enumerate(self.names) if name != name_remove]
        self.names = [self.names[i] for i in idx]
        self.encodes = self.encodes[idx]
        self.save_data_pkl()
        
    def gen_data_encode(self):
        
        self.names = []
        self.encodes = []
        for path_dir in glob.glob(os.path.join(self.path_data_base,"*")):
            name = os.path.basename(path_dir)
            for image_path in glob.glob(os.path.join(path_dir , "*")):
                print(f"loading file: {image_path}")
                image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
                face = model_test[0].get_bbox(image)
                if len(face) !=1:
                    print(f"File {image_path} is can't use to recognize ")
                    continue
                box  = xywh2xyxy(face)[0]
                self.names.append(name)
                self.encodes.append(face_encode(image,box,mode = "rgb"))
        self.encodes = np.array(self.encodes)
        self.save_data_pkl()

    def save_data_pkl(self,file_path=None):
        if file_path == None:
            file_path= self.encode_data_path

        with open(file_path, "wb") as f:
            pickle.dump(self.names, f)
            pickle.dump(self.encodes, f)

    def _get_face_encodes(self,image,boxes_face = None):
        if boxes_face == None:
            pass

        encode = face_encode(image,boxes_face[0])
        return boxes_face,[encode]
    def draw_face(self,frame,location,name):

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 2

        x1,y1,x2,y2 = location
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        frame = cv2.putText(frame, name, (x1, y1), font, fontScale, color, thickness, cv2.LINE_AA)
        return frame
    
    def face_distance(self,face_encodings, face_to_compare):
        """
        Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
        for each comparison face. The distance tells you how similar the faces are.

        :param faces: List of face encodings to compare
        :param face_to_compare: A face encoding to compare against
        :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
        """
        if len(face_encodings) == 0:
            return np.empty((0))

        return np.linalg.norm(face_encodings - face_to_compare, axis=1)
    def check_face(self,frame,face_know,draw_image = True,threshold_value = 0.8):
        names = []
        frame_copy = cv2.cvtColor(frame.copy(),cv2.COLOR_BGR2RGB)
        locations,face_encodes = self._get_face_encodes(frame_copy,face_know)
        
        for location, face_encode in zip(locations, face_encodes):
            face_distances = self.face_distance(self.encodes, face_encode)
            index = np.argmin(face_distances)
            value = face_distances[index]
            name = "unknow"
            if value < threshold_value:
                name = self.names[index]
            names.append(name)
            if draw_image:
                location = [int(x) for x in location]
                frame = self.draw_face(frame,location,name)
        return names,frame

import cv2


  # Example: Draw a white filled circle as a mask

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    vertificat = FacialVertification()
    
    while True:
        begint = time.time()
        # Capture frame-by-frame
        ret, frame = cam.read()
        img,_ = detect_liveness(frame,vertificat,visulize=True)
        cv2.imshow("frame",img)
        if cv2.waitKey(8)==ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
            