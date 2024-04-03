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
from utility import *
from resources.classification_model.train_pipeline import init_and_train_model_from_scatch_pipeline
import faiss
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

SPLIT = "()()()()()()()"
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
       self.from_cache()
       getattr(self,config['data']['load_data_type'])()
       self.label_setup()
       self.knn_setup()
       self.type_classifier = "knn_distance"
       if config["autofaiss"]['use']:
           print("Use autofaiss")
           self.type_classifier = "autofaiss_distance"
           self.auto_faiss_setup()
       if config['classifier']['use']:
           print("Use deep learning classifier")
           self.train()
    
    def auto_faiss_setup(self):
        faiss_config = config["autofaiss"]
        save_folder = faiss_config['root_folder']
        print(f"Saving faiss encode at {save_folder}")
        embedding_folder = os.path.join(save_folder,"embeddings") 
        index_folder = os.path.join(save_folder,"index_folder")
        os.makedirs(embedding_folder,exist_ok=True)
        os.makedirs(index_folder,exist_ok=True)
        np.save(f"{embedding_folder}/part1.npy", self.encodes)
        os.system(f'autofaiss build_index --embeddings="{embedding_folder}" --index_path="{index_folder}/knn.index" --index_infos_path="{index_folder}/index_infos.json" --metric_type="l2"')
        self.my_index = faiss.read_index(glob.glob(f"{index_folder}/*.index")[0])
    def knn_setup(self):
        self.knn = NearestNeighbors(n_neighbors=3, algorithm='brute')
        self.knn.fit(self.encodes)
        self.threshsold = config['face_recognition']['threshold']
    def recognition(self,frame,return_id =None):
        '''
        return name,bbox(x1,y1,x2,y2)
        or 
        return name,bbox ,id 
        '''
        face = face_detect(frame)[0]
        if len(face) ==0:
            return None,None
        #1,feature_dims
        feature = np.expand_dims(self.extract_feature(frame,face),axis = 0)
        if self.type_classifier == "knn_distance":
            distances, indices = self.knn.kneighbors(feature)
        elif self.type_classifier == "autofaiss_distance":
            k = 3
            distances, indices = self.my_index.search(feature, k)
        count = self.get_best(distances=distances,indices=indices)
        
        if not len(count):
            return "unknow",face
        id =  count.most_common(1)[0][0]
        result = [self.get_name(id),face]
        if return_id:
            result.append(id)
        return  result
    def get_best(self,distances,indices):
        ids = [self.label2id[label] for i,label in enumerate(indices[0]) if distances[0][i] <self.threshsold]
        print(ids)
        name_counts = collections.Counter(ids)
        return name_counts
    def get_name(self,id):
        return self.id2name[id]
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
    
    # def from_pkl(self):
    #     with open("resources/face_encode_data/dlib.pkl", "rb") as f:
    #         self.id = pickle.load(f)
    #         self.encodes = pickle.load(f)
    def label_setup(self):
        assert(self.id!= None)
        self.label2id = {}
        count = 0
        for id in self.id:
            if not self.label2id.get(id,None):
                self.label2id[count] = id
                count +=1
            
    def from_image(self):
        self.id = []
        self.encodes = []
        self.label_index = []
        self.id2name = {}
        for i,path_dir in enumerate(glob.glob(os.path.join(config['data']['data_path'],"*"))):
            
            try:
                int(os.base_name(path_dir.spit("-->")[0]))
            except:
                base_name = str(i) +SPLIT+os.path.basename(path_dir).split(SPLIT)[-1]
                dir_name = os.path.dirname(path_dir)
                new_name = os.path.join(dir_name,base_name)
                os.rename(path_dir,new_name)
                
        for i,path_dir in enumerate(glob.glob(os.path.join(config['data']['data_path'],"*"))):
            name = os.path.basename(path_dir)
            id = name.split(SPLIT)[0]
            name = name.split(SPLIT)[-1]
            self.id2name[id] = name
            for image_path in glob.glob(os.path.join(path_dir , "*")):
                if self.cache.get(image_path) is None:
                    print(f"loading file: {image_path}")
                    image  = cv2.imread(image_path)
                    feature = self.extract_feature(image)
                    self.cache[image_path] = feature
                feature  = self.cache[image_path]
                self.label_index.append(i)
                self.id.append(id)
                self.encodes.append(feature)
        self.label_index = np.array(self.label_index)
        self.encodes = np.array(self.encodes)
    
        if config['data'].get('save_data',None):
            self.save_data()
                
            
    def save_data(self):
        self.save_data_pkl()
    
    def get_face_distance(self,feature):
        return getattr(utility,config['face_recognition']['face_distance'])(self.encodes,feature)
    def from_cache(self):
        self.cache = {}
        if os.path.exists(config['data']['cache']):
            with open(config['data']['cache'],"rb") as f:
                self.cache = pickle.load(f)
    def save_data_pkl(self):
        print("save data pkl")
        file_path = os.path.join("resources/face_encode_data","dlib.pkl")
        if os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs("resources/face_encode_data",exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.cache,f)
    def train(self):
        train_config = config['classifier']
        num_class  = max(self.label_index)+1
        feature_dims,dropout = train_config['net']['feature_dims'] ,train_config['net']['dropout']                         
        device = "cuda" if torch.cuda.is_available() else "cpu"
        max_epochs = train_config['max_epochs']
        batch_size = train_config['batch_size']
        caculumn_size = train_config['caculumn_size']
        self.classifier = init_and_train_model_from_scatch_pipeline((self.encodes,self.label_index),
                                                                    num_class=num_class,
                                                                    feature_dims=feature_dims,
                                                                    device = device,
                                                                    max_epochs=max_epochs,
                                                                    batch_size= batch_size,
                                                                    caculumn_size = caculumn_size,
                                                                    dropout=dropout)
        
def check_box(box):
    x1,y1,x2,y2 = box
    if (x2-x1)*(y2-y1) < 30:
        return False
    return True
if __name__ == "__main__":
    
    cam = cv2.VideoCapture(0)
    recog = Recognition()
    while True:
        ret,frame = cam.read()
        name,box = recog.recognition(frame)
        if box and check_box(box):
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