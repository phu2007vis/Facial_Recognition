from phuoc_uitls import *
import random
import questions
from collections import defaultdict
from src.anti_spoof_predict import spoof_predict
def crop_center(image, crop_width = 350, crop_height = 350):
    """
    Crop the center region of an image.

    Parameters:
    - image: Input image (NumPy array).
    - crop_width: Width of the cropped region.
    - crop_height: Height of the cropped region.

    Returns:
    - Cropped image.
    """
    h, w = image.shape[:2]
    start_w = (w - crop_width) // 2
    start_h = (h - crop_height) // 2
    cropped_image = image[start_h:start_h + crop_height, start_w:start_w + crop_width ,:]
    return cropped_image
class SuperDetect:
    def __init__(self,cam = None,opt_path = r"D:\face_liveness_detection-Anti-spoofing\config.yaml"):
        self.opt = read_yaml(opt_path)
        if cam is None:
            self.cam = cv2.VideoCapture(0)
        else:
            self.cam = cam
        self.limit_questions = self.opt['limit_questions']
        self.face_verti = FacialVertification()
    def reset(self):
        self.index_question = 0
        self.question = None
        self.frame = None
        self.name = None
        self.frames = []
        self.counter_ok_questions = 0
        self.error = 0
    def update_question(self):
        self.index_question +=1
        self.index_question = self.index_question%2
        self.question = questions.question_bank(self.index_question)
        self.counter_ok_consecutives = 0
        self.ok = False
    def update_frame(self):
        self.ret,self.frame =  self.cam.read()
        if self.question:
            self.frame = crop_center(self.frame)
            # mask = np.zeros_like(self.frame)
            # cv2.circle(mask, (self.frame.shape[1] // 2, self.frame.shape[0] // 2), 150, (255, 255, 255), thickness=-1)
            # self.frame = cv2.bitwise_and(self.frame, mask)
            im = put_text(self.frame.copy(),f"{self.counter_ok_questions}\\{self.limit_questions} {self.question}")
        cv2.imshow('liveness_detection',im)
        cv2.waitKey(1)
    def update_output(self):
        self.update_frame()
        self.output = detect_liveness(self.frame)
        self.check_ouput()
        
        if self.counter_ok_consecutives == self.opt['limit_consecutives']:
            self.counter_ok_questions+=1
            self.ok = True
    def check_ouput(self):
        face_info = self.output[1]
        self.result = questions.challenge_result(self.question,face_info)
        if self.result == "pass":
            if len(face_info['boxes_faces'])==1 and len(self.frames)< self.opt['max_frames_save']:
                self.frames.append([self.frame.copy(),face_info])
            
            self.counter_ok_consecutives +=1
    def random_select(self,mode = "vertification" ):
        if mode == "vertification":
            return random.sample(self.frames, self.opt['num_faces_check'])
        elif mode == "register":
            return random.sample(self.frames, self.opt['num_faces_register'])
    def face_register(self,name):
        labels = defaultdict(int)
        encoding = []
        name2,label = self.face_vertification()
        if name2 != "unknow":
            return "face is already exists",False
        for image,face_info in self.random_select():
            box = face_info['boxes_faces']
            encoding.append(self.face_verti._get_face_encodes(image,box)[1][0])
            label = face_info['label'][0]
            labels[label]+=1
        self.label = max(labels,key=labels.get)
        if self.label == "real":
            return self.face_verti.register_new_people(name,encoding)
        return "you are fake",False
    def face_vertification(self):
        names = defaultdict(int)
        labels = defaultdict(int)
        for image,face_info in self.random_select():
            box = face_info['boxes_faces']
            label =  face_info['label'][0]
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            names_detect,image = self.face_verti.check_face(image,box)
            labels[label]+=1
            for name in names_detect:
                names[name] += 1  
        self.name = max(names, key=names.get)
        self.label = max(labels,key=labels.get)
        return self.name,self.label
    def detect_liveness(self,process_main):
        self.process_main = process_main
        self.reset()
        while self.counter_ok_questions <= self.opt['limit_questions']:
            self.update_question()
            for _ in range(self.opt['limit_try']):
                self.update_output()
                if self.ok:
                    break
            if not self.ok:
                self.counter_ok_questions  = max(0,self.counter_ok_questions - 1)
                self.error +=1
            if  self.error > self.opt['max_error']:
                cv2.destroyAllWindows()
                return False
        cv2.destroyAllWindows()
        return True

if __name__  == "__main__":
    model = SuperDetect()

# while True:
#     oke  = model.detect_liveness()
#     if oke :
#         model.face_vertification()
#         if model.name == 'unknow':
#             color =  (0,0,255)
#             text2  = f"You not registry"
#         elif model.label!=1:
#             color =  (0,0,255)
#             text2  = f"You are cheating"
#         else:
#             color = (0,255,0)
#             text2 = f"Welcome {model.name}"
          
#     else:
#         model.name = None
#         text2 = None
#         color = (0,0,255)

    # while True:
    #     text1 = f"Successful liveness" if oke else "Fail liveness"
    #     frame = model.cam.read()[1]
    #     frame = put_text(frame,text = text1,color=color)
    #     if text2:
    #         frame = put_text(frame,text = text2,color=color,y = 50)
    #     cv2.imshow("liveness_detection",frame)
    #     if cv2.waitKey(8) == ord("q"):
    #         break
    
    
cv2.destroyAllWindows()
                
    
        
                