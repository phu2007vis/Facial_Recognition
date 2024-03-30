from sklearn.neighbors import NearestNeighbors
import cv2
import numpy as np
import math
from utility import xywh2xyxy
import yaml
import pickle
import os
import glob
import utility
import collections
with open("config.yaml","r") as f:
    config = yaml.safe_load(f)
if config['type_recognition'] == "dlib":
    import dlib
    landmark_detector = dlib.shape_predictor(r"resources/dlib/shape_predictor_68_face_landmarks.dat")
    feature_extractor = dlib.face_recognition_model_v1(r"resources/dlib/dlib_face_recognition_resnet_model_v1.dat")
    detector = dlib.get_frontal_face_detector()
    from anti_spoof_predict import model_test,spoof_predict
else:

    from facenet_pytorch import InceptionResnetV1
    import torch
    resnet = InceptionResnetV1(pretrained='casia-webface').eval()
    
name_real_or_fake = ['fake','real','fake']

class Detection:
    def __init__(self):
        caffemodel = "./resources/detection_model/Widerface-RetinaFace.caffemodel"
        deploy = "./resources/detection_model/deploy.prototxt"
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
                                   out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
        bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
        #x1,y1,w,h
        return bbox
det = Detection()
def face_detect(frame):
    '''
    return [[x1,y1,x2,y2],[...],...]
    '''
    return xywh2xyxy([det.get_bbox(frame)])

class Recognition:
    def __init__(self) -> None:
       getattr(self,config['data']['load_data_type'])()
       self.label_setup()
       self.knn_setup()
    def knn_setup(self):
        self.knn = NearestNeighbors(n_neighbors=3, algorithm='brute')
        self.knn.fit(self.encodes)
    def recognition(self,frame):
        face = face_detect(frame)[0]
        if len(face) ==0:
            return None,None
        feature = self.extract_feature(frame,face)
        
        distances, indices = self.knn.kneighbors(np.expand_dims(feature,axis = 0))
        names = [self.label2name[label] for i,label in enumerate(indices[0]) if distances[0][i] < config['face_recognition']['threshold']]
        name_counts = collections.Counter(names)
        if not len(name_counts):
            return "unknow",face
        return name_counts.most_common(1)[0][0],face
        # distance = self.get_face_distance(feature)
        # name = "unknow"
        # index = np.argmin(distance)
        # if distance[index] < config['face_recognition']['threshold']:
        #     name = self.label2name[self.label[index]]
        return name,face
    def dlib(self,frame,face):
        '''
        frame : bgr
        '''

        face_dlib = dlib.rectangle(*face)
        shape = landmark_detector(frame, face_dlib)
        return feature_extractor.compute_face_descriptor(frame, shape)
    
    def extract_feature(self,frame,face = None):
        '''
        frame : bgr 
        face  : x1,y1,x2,y2
        '''
        if face is None:
            face = face_detect(frame)[0]
        return np.array(getattr(self,config['type_recognition'])(frame,face))
    
    def from_pkl(self):
        with open(config['data']['data_path'], "rb") as f:
            self.names = pickle.load(f)
            self.encodes = pickle.load(f)
    def label_setup(self):
        assert(self.names!= None)
        self.label2name = {}
        self.name2label = {}
        self.label = []
        count = 0
        for name in self.names:
            if not self.label2name.get(name,None):
                self.label2name[count] = name
                self.name2label[name]  = count
                count +=1
            self.label.append(self.name2label[name])
    def from_image(self):
        self.names = []
        self.encodes = []
        self.names = []
        self.encodes = []
        for path_dir in glob.glob(os.path.join(config['data']['data_path'],"*")):
            name = os.path.basename(path_dir)
            for image_path in glob.glob(os.path.join(path_dir , "*")):
                print(f"loading file: {image_path}")
                image  = cv2.imread(image_path)
                feature = self.extract_feature(image)
                self.names.append(name)
                self.encodes.append(feature)
        self.encodes = np.array(self.encodes)
        if config['data'].get('save_data',None):
            self.save_data()
            
    def save_data(self):
        self.save_data_pkl()
    
    def get_face_distance(self,feature):
        return getattr(utility,config['face_recognition']['face_distance'])(self.encodes,feature)
    
    def save_data_pkl(self):
        file_path = os.path.join("resources/face_encode_data",config['type_recognition']+".pkl")
        os.makedirs("resources/face_encode_data",exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.names, f)
            pickle.dump(self.encodes, f)
if __name__ == "__main__":
    from utility import *
    cam = cv2.VideoCapture(0)
    recog = Recognition()
    while True:
        ret,frame = cam.read()
        name,box = recog.recognition(frame)
        if box:
            color = (0,255,0)
            if name=="unknow":
                color = (0,0,255)
            frame = draw_faces(frame,[box])
            frame = put_text(frame,name,y= 100,color = color)
            label,_  =  spoof_predict(frame,[box])
            color = (0,0,255)
            if name_real_or_fake[label] == "real":
                color = (0,255,0)
            frame = put_text(frame,name_real_or_fake[label],y= 50,color=color)
            
        cv2.imshow("farme",frame)
        #q to exit
        if cv2.waitKey(8) == ord("q"):
            break











# stream_url = "http://192.168.0.100:5000/image_feed"
# def receive_frames():
#     # Send a GET request to the Flask server's video feed endpoint
#     response = requests.get(stream_url, stream=True)
#     if response.status_code == 200:
#         # Read the raw bytes of the response
#         bytes_data = bytes()
#         for chunk in response.iter_content(chunk_size=1024):
#             bytes_data += chunk
#             # Check for frame boundary
#             a = bytes_data.find(b'\xff\xd8')
#             b = bytes_data.find(b'\xff\xd9')
#             if a != -1 and b != -1:
#                 # Extract frame and reset bytes_data
#                 frame_data = bytes_data[a:b+2]
#                 bytes_data = bytes_data[b+2:]
#                 # Decode JPEG frame
#                 frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
#                 yield frame