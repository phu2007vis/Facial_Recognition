from resources.detection_model.detect import face_detect,mtcnn_face_detect
import dlib
import os.path as osp
import os
import cv2
import numpy as np
import skimage
from resources.utility import check_box,get_annotation_map
import face_alignment
import torch
from resources.folder_utils.utils import count_file
from PIL import Image
from tqdm import tqdm

# landmark_detector = dlib.shape_predictor(r"resources/dlib/shape_predictor_68_face_landmarks.dat")
# feature_extractor = dlib.face_recognition_model_v1(r"resources/dlib/dlib_face_recognition_resnet_model_v1.dat")

device =  "cuda" if torch.cuda.is_available() else "cpu"
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False,device = device)
def get_dlib_full_object_detections(image,bboxes,type= "dlib"):
    box = bboxes[0]
    if type == "dlib":
        face_dlib = dlib.rectangle(*box)
        shape = landmark_detector(image, face_dlib)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    else:
        face_dlib =  dlib.rectangle(*box)
        landmarks = fa.get_landmarks(image,detected_faces =bboxes )[0]
    dlib_points = [dlib.point(landmarks[i, 0], landmarks[i, 1]) for i in range(len(landmarks))]
    return dlib.full_object_detections([dlib.full_object_detection(face_dlib, dlib_points)])
def extract_feature(image,dlib_full_object_detections ):
    return feature_extractor.compute_face_descriptor(image,dlib_full_object_detections)
def extract_feature_pipeline(image,bboxes:list = None,type = "dlib"):
    '''
    numpy array with shape (128,)
    '''
    if bboxes is None:
        bboxes = face_detect(image)
    assert(len(bboxes)==1)
    dlib_full_object_detections = get_dlib_full_object_detections(image,bboxes,type)
    return np.array(extract_feature(image,dlib_full_object_detections)).squeeze()
def extract_landmarks(image,bboxes,type = "dlib",return_dlib_all = False):
    '''
    bbox = [[x1,y1,x2,y2]]
    '''
    if type == "dlib":
        box  = bboxes[0]
        face_dlib = dlib.rectangle(*box)
        shape = landmark_detector(image, face_dlib)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    elif type=="face_alignment":
        face_dlib = bboxes[0]
        face_dlib =  dlib.rectangle(*face_dlib)
        landmarks = fa.get_landmarks(image,detected_faces =bboxes )[0]
    return landmarks
def pad_image(image, padding=70):

    image = Image.fromarray(image)
    width, height = image.size
    padded_image = Image.new(image.mode, (width + padding, height + padding), (0, 0, 0))
    padded_image.paste(image, (0, 0))
    return np.asarray(padded_image)

def extract_face_from_image(image,image_size = 224,type_landmakrs = "dlib"):
    image = pad_image(image,20)
    
    bboxes = face_detect(image)
    if not check_box(bboxes[0]):
        bboxes = mtcnn_face_detect(image)
        if not check_box(bboxes[0]):
            return None
    x1,y1,x2,y2 = bboxes[0]
    landmarks = extract_landmarks(image,bboxes,type=type_landmakrs,return_dlib_all=True)
    outline = landmarks[[*range(17), *range(26,16,-1)]]
   
    Y, X = skimage.draw.polygon(outline[:,1], outline[:,0])
    cropped_img = np.zeros(image.shape, dtype=np.uint8)
    cropped_img[Y, X] = image[Y, X]
    cropped_img = cropped_img[y1:y2,x1:x2,:]
    cropped_img = cv2.resize(cropped_img,(image_size,image_size))
    return cropped_img
def extrace_face(input_folder,output_folder,image_size = 224,type_landmarks = "dlib",logfile = "log.txt"):
    processing_bar = tqdm(range(count_file(input_folder)))
    for person_name in os.listdir(input_folder):
        sub_output_folder = os.path.join(output_folder,person_name)
        sub_input_folder = os.path.join(input_folder,person_name)
        
        os.makedirs(sub_output_folder,exist_ok=True)
        if len(os.listdir(sub_input_folder))==0:
            print(f"{input_folder} is empty")
        for file_name in os.listdir(sub_input_folder):
            processing_bar.update(1)
            file_path = osp.join(sub_input_folder,file_name)
            image = cv2.imread(file_path)
           
        
            cropped_img = extract_face_from_image(image,image_size,type_landmarks)
            if cropped_img is None:
                with open(logfile, "a+") as f:
                    f.write(f"Important {file_path} can not extract face please veritify this face ! \n")
                continue
            output_file = osp.join(sub_output_folder,file_name)
            cv2.imwrite(output_file,cropped_img)

        
            



