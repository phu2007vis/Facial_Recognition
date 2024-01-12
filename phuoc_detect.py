from phuoc_uitls import *
import random
import questions
from collections import defaultdict
class SuperDetect:
    def __init__(self,opt_path = r"D:\face_liveness_detection-Anti-spoofing\config.yaml"):
        self.opt = read_yaml(opt_path)
        self.cam = cv2.VideoCapture(0)
        self.limit_questions = self.opt['limit_questions']
        self.face_verti = FacialVertification()
    def reset(self):
        self.question = None
        self.frame = None
        self.name = None
        self.frames = []
        self.counter_ok_questions = 0
        self.error = 0
    def update_question(self):
        index_question = random.randint(0,len(questions.questions)-1)
        self.question = questions.question_bank(index_question)
        self.counter_ok_consecutives = 0
        self.ok = False
    def update_frame(self):
        self.ret,self.frame =  self.cam.read()
        if self.question:
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
        self.result = questions.challenge_result(self.question,self.output)
        if self.result == "pass":
            if len(self.output['boxes_faces'])==1 and len(self.frames)< self.opt['max_frames_save']:
                self.frames.append([self.frame.copy(),self.output['boxes_faces']])
            self.counter_ok_consecutives +=1
    def random_select(self):
        return random.sample(self.frames, self.opt['num_faces_check'])
    def face_vertification(self):
        names = defaultdict(int)
        for image,box in self.random_select():
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            names_detect,image = self.face_verti.check_face(image,box)
            for name in names_detect:
                names[name] += 1  
        self.name = max(names, key=names.get)
        

    
    def detect_liveness(self):
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
                return False
        return True

model = SuperDetect()
oke  = model.detect_liveness()
import time
t = time.time()
model.face_vertification()
print(time.time()-t)
while True:
    text = f"welcome {model.name}" if oke else "fail"
    frame = model.cam.read()[1]
    color = (0,255,0) if oke else (0,0,255)
    frame = put_text(frame,text = text,color=color)
    cv2.imshow("liveness_detection",frame)
    if cv2.waitKey(8) == ord("q"):
        break
    
cv2.destroyAllWindows()
                
    
        
                