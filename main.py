import os.path
import datetime
import pickle
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
from phuoc_detect import SuperDetect
import util

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+10+10")

        # Initialize self.cap before calling add_webcam
        self.cap = cv2.VideoCapture(0)

        self.login_button_main_window = util.get_button(self.main_window, 'login', 'green', self.login)
        self.login_button_main_window.place(x=750, y=200)

        self.logout_button_main_window = util.get_button(self.main_window, 'remove face', 'red', self.logout)
        self.logout_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(self.main_window, 'register new user', 'gray',
                                                                    self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'
        
        # Initialize main_detect after self.cap is created
        self.main_detect = SuperDetect(cam=self.cap)

    def add_webcam(self, label):
        self._label = label
        self.display_mode = 1
        self.process_webcam()

    def process_webcam(self):
        if self.display_mode == 1:
            ret, frame = self.cap.read()
            self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)
        self._label.after(20, self.process_webcam)

    def login(self):
        if self.main_detect.detect_liveness(self):
            name,label = self.main_detect.face_vertification()
            if label == "real":
                if name == "unknow":
                    util.msg_box("Fail!","Can't detect face")
                    
                else:
                    self.face_loging = name
                    util.msg_box("Successfull!",f"Welcome {name}")
            else:
                util.msg_box("Fail!","You are fake")

    def logout(self):
        if self.face_loging is not None:
            self.main_detect.face_verti.remove_face(self.face_loging)
            util.msg_box("Successfull!",f"Removed {self.face_loging}")
            self.face_loging = None

    def register_new_user(self):

        if self.main_detect.detect_liveness(self):
            
            self.register_new_user_window = tk.Toplevel(self.main_window)
            self.register_new_user_window.geometry("1200x520+10+10")
            
            self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user)
            self.try_again_button_register_new_user_window.place(x=750, y=400)

            self.capture_label = util.get_img_label(self.register_new_user_window)
            self.capture_label.place(x=10, y=0, width=700, height=500)


            self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
            self.entry_text_register_new_user.place(x=750, y=150)

            self.text_label_register_new_user = util.get_text_label(self.register_new_user_window, 'Please, \ninput username:')
            self.text_label_register_new_user.place(x=750, y=70)
    
            self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user)
            self.accept_button_register_new_user_window.place(x=750, y=300)
     
            
    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def start(self):
        self.main_window.mainloop()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")
        mess,oke = self.main_detect.face_register(name)
        if oke:
            util.msg_box("Susscessfuly!",mess)
        else:
            util.msg_box("Fail!",mess)
        self.register_new_user_window.destroy()

if __name__ == "__main__":
    app = App()
    app.start()
