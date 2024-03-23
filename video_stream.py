from datetime import datetime
import yaml
import cv2

with open("config.yaml","r") as f:
    config = yaml.safe_load(f)
    
class Stream():
    def __init__(self):
        self.vid = cv2.VideoCapture(config['stream_uri'])
        
    def get_frame(self):

        if not self.vid.isOpened():
            return

        while True:
            _, img = self.vid.read()
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            img = cv2.putText(img, datetime.now().strftime("%H:%M:%S"), org, font,
                              fontScale, color, thickness, cv2.LINE_AA)
            self.image = img

            yield cv2.imencode('.jpg', img)[1].tobytes()#frame