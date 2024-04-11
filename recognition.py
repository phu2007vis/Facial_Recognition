from sklearn.neighbors import NearestNeighbors
import cv2
from resources.detection_model.detect import face_detect
import numpy as np
from resources.sql.sql_contronler import *
import pickle
import os
import glob
import utility
import collections
from resources.utility import *
import faiss
from anti_spoof_predict import spoof_predict
import dlib
from resources.utility import check_box
import gc
import asyncio
import threading
import time


landmark_detector = dlib.shape_predictor(r"resources/dlib/shape_predictor_68_face_landmarks.dat")
feature_extractor = dlib.face_recognition_model_v1(r"resources/dlib/dlib_face_recognition_resnet_model_v1.dat")


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
           from resources.classification_model.train_pipeline import init_and_train_model_from_scatch_pipeline
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
        del self.encodes
        gc.collect()
        self.my_index = faiss.read_index(glob.glob(f"{index_folder}/*.index")[0])
    def knn_setup(self):
        self.knn = NearestNeighbors(n_neighbors=3, algorithm='brute')
        self.knn.fit(self.encodes)
        self.threshsold = config['face_recognition']['threshold']
    def from_sql(self):
        results = get_all_features_and_labels()
        if results.pop(-1):
            print("Not contain any picture")
        self.encodes,self.id = results
        self.id2name = get_id2name()[0]
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
            k = 4
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
        ids = [self.id[label] for i,label in enumerate(indices[0]) if distances[0][i] <self.threshsold]
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


import time

check_in_queue = []
capture_thread = None

def my_async_function():
    global check_in_queue
    copy_check_in_queue = check_in_queue[:]
    check_in_queue = [copy_check_in_queue[-1]]
    for id, current_time, time_str in copy_check_in_queue:
        try:
            check_in(id, current_time, time_str)
        except Exception as e:
            pass
            

def check_in_loop():
    global check_in_queue
    while True:
        print(len(check_in_queue))
        if len(check_in_queue):
            my_async_function()
        time.sleep(5)

def capture_and_process(recog):
    global check_in_queue
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        try:
            name, box, id = recog.recognition(frame, return_id=True)
        except Exception as e:
            print(e)
            continue
        current_time = get_time()  # Assuming get_time() is defined elsewhere
        if not check_in_queue or check_check_in(check_in_queue[-1], (id, current_time, str(time.time()))):
            check_in_queue.append((id, current_time, str(time.time())))
        if box and check_box(box):
            color = (0, 255, 0)
            if name == "unknown":
                color = (0, 0, 255)
            frame = draw_faces(frame, [box])  # Assuming draw_faces() is defined elsewhere
            frame = put_text(frame, name, y=100, color=color)  # Assuming put_text() is defined elsewhere
            label, _ = spoof_predict(frame, [box])  # Assuming spoof_predict() is defined elsewhere
            color = (0, 0, 255)
            if name_real_or_fake[label] == "real":
                color = (0, 255, 0)
            frame = put_text(frame, name_real_or_fake[label], y=50, color=color)  # Assuming name_real_or_fake is defined elsewhere
        cv2.imshow("frame", frame)
        if cv2.waitKey(8) == ord("q"):
            break

def run_capture_and_process(recog):
    global capture_thread
    capture_thread = threading.Thread(target=capture_and_process, args=(recog,))
    capture_thread.start()
    check_in_thread = threading.Thread(target=check_in_loop)
    check_in_thread.start()

if __name__ == "__main__":
    recog = Recognition()  # Assuming Recognition is defined elsewhere
    run_capture_and_process(recog)










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