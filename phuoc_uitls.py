import dlib
from profile_detection import f_detector
from emotion_detection import f_emotion_detection
import cv2
import numpy as np
import dlib         # 人脸识别的库 Dlib
import cv2          # 图像处理的库 OpenCV
from face_recognition_models import pose_predictor_model_location,face_recognition_model_location
face_detector            = dlib.get_frontal_face_detector()
profile_detector         = f_detector.detect_face_orientation()
emotion_detector         = f_emotion_detection.predict_emotions()
predictor = dlib.shape_predictor(pose_predictor_model_location())
face_reco_model = dlib.face_recognition_model_v1(face_recognition_model_location())

def face_encode(img_rd,face):

    x1,y1,x2,y2 = face
    assert(x1<x2 and y1 <y2)
    face_shape = predictor(img_rd,dlib.rectangle(left = x1,right = x2,top = y1,bottom = y2))
    face_desc = face_reco_model.compute_face_descriptor(img_rd, face_shape)
    return np.asarray(face_desc)
    

def get_areas(boxes):
    areas = []
    for box in boxes:
        x0,y0,x1,y1 = box
        area = (y1-y0)*(x1-x0)
        areas.append(area)
    return areas

def convert_rectangles2array(rectangles,image):
    '''
    output : [[y1,x1,y2,x2]*num_faces]
    '''
    res = np.array([])
    for box in rectangles:
        [x0,y0,x1,y1] = max(0, box.left()), max(0, box.top()), min(box.right(), image.shape[1]), min(box.bottom(), image.shape[0])
        new_box = np.array([x0,y0,x1,y1])
        if res.size == 0:
            res = np.expand_dims(new_box,axis=0)
        else:
            res = np.vstack((res,new_box))
    return res


def get_face(image):
    '''
    image : rgb (w,h,3)
    '''
    if len(image.shape)==3:
        image  = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    rectangles = face_detector(image, 0)
    boxes = convert_rectangles2array(rectangles,image)
    return boxes

def draw_faces(image,boxes):
    for box in boxes:
        x1,y1,x2,y2 = box
        image = cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 1)
    return image

def put_text(im,text,y = 0,color = (0,0,255)):

    #im = cv2.flip(im, 1)
    cv2.putText(im,text,(10,50+y),cv2.FONT_HERSHEY_COMPLEX,1,color,2)
    return im

def detect_liveness(im):
    # preprocesar data
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    boxes_face = get_face(gray)

    if len(boxes_face)!=0:
        
        boxes_face = [list(boxes_face[0])]

        # -------------------------------------- emotion_detection ---------------------------------------
        '''
        input:
            - imagen RGB
            - boxes_face: [x1,y1,x2,y2]
        output:
            - status: "ok"
            - emotion: ['happy'] or ['neutral'] ...
            - box: [[579, 170, 693, 284]]
        '''
        _,emotion = emotion_detector.get_emotion(im,boxes_face)
        # -------------------------------------- blink_detection ---------------------------------------
        '''
        input:
            - imagen gray
            - rectangles
        output:
            - status: "ok"
            - COUNTER: # frames consecutivos por debajo del umbral
            - TOTAL: # de parpadeos
        '''
        # COUNTER,TOTAL = blink_detector.eye_blink(gray,rectangles,COUNTER,TOTAL)
    else:
        boxes_face = []
        emotion = []

    # -------------------------------------- profile_detection ---------------------------------------
    '''
    input:
        - imagen gray
    output:
        - status: "ok"
        - profile: ["right"] or ["left"]
        - box: [[579, 170, 693, 284]]
    '''
    box_orientation, orientation = profile_detector.face_orientation(gray)

    # -------------------------------------- output ---------------------------------------


    output = {
        'emotion': emotion,
        'orientation': orientation,
        'boxes_faces': boxes_face,
        
    }
    # for i,key in enumerate(output.keys()):
    #     value = output[key]
    #     if len(value)==0:
    #         text = key+": "+"None"
    #     else:
    #         text = key+": "+str(value[0])

    #     im = put_text(im,text,i*30)
    return output

def read_yaml(yaml_path):
    import yaml

    # Load YAML from a file
    with open(yaml_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data
import pickle
import os
import glob
class FacialVertification:
    def __init__(self,encode_data_path = r"D:\face_liveness_detection-Anti-spoofing\data.pkl",path_data_base  = r"D:\face_liveness_detection-Anti-spoofing\DataBase"):

        self.encode_data_path = encode_data_path
        self.path_data_base = path_data_base
        self.names,self.encodes = self._load_data_face(encode_data_path)
        # self.gen_data_encode()

    def _load_data_face(self,face_data_path ):

        with open(face_data_path, "rb") as f:
            names = pickle.load(f)
            encodes = pickle.load(f)
        return names,encodes

    def gen_data_encode(self):
        

        self.names = []
        self.encodes = []
        for path_dir in glob.glob(os.path.join(self.path_data_base,"*")):
            
            name = os.path.basename(path_dir)
            self.names += [name] * len(glob.glob(os.path.join(path_dir , "*")))
            for image_path in glob.glob(os.path.join(path_dir , "*")):
                print(f"loading file: {image_path}")
                image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
                gray  = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
                boxes_face = get_face(gray)
                boxes_face = [list(boxes_face[0])]
                self.encodes.append(face_encode(image,boxes_face[0]))
        self.save_data_pkl()

    # def add_new_image(self,frame,name : str):
    #     locations = face_recognition.face_locations(frame)
    #     if len(locations)==1:
    #         face_encoding = face_recognition.face_encodings(frame,known_face_locations=locations,num_jitters = 2)[0]
    #         self.encodes.append(face_encoding)
    #         self.names.append(name)
    #         return True
    #     else:
    #         return False


    def save_data_pkl(self,file_path=None):
        if file_path == None:
            file_path= self.encode_data_path

        with open(file_path, "wb") as f:
            pickle.dump(self.names, f)
            pickle.dump(self.encodes, f)

    def _get_face_encodes(self,image,boxes_face = None):
        if boxes_face == None:
            gray  = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            boxes_face = get_face(gray)
            boxes_face = [list(boxes_face[0])]
        if len(boxes_face)==4:
            boxes_face = [boxes_face]
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
    def check_face(self,frame,face_know,draw_image = True,threshold_value = 0.45):
        names = []
        locations,face_encodes = self._get_face_encodes(frame,face_know)
        
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