
import numpy as np
from resources.mivolo.model.mi_volo import MiVOLO
from resources.detection_model.detect import face_detect
import gdown
import os
from resources.utility import check_box


url = 'https://drive.google.com/uc?id=17xOIu7eEgpFoIwDB9Ez_k3qGgICcRN65'

# Define the output file path
checkpoint = 'resources/mivolo/model_imdb_age_gender_4.22.pth'
# Change the name and extension according to the file type
if not os.path.exists(checkpoint):
    os.system("pip install --upgrade gdown")
    gdown.download(url, checkpoint, quiet=False,use_cookies = False)

class Predictor:
    def __init__(self, verbose: bool = False,device ="cpu"):
        
        self.age_gender_model = MiVOLO(
            checkpoint,
            device = device,
            half=True,
            use_persons=False,
            disable_faces=False,
            verbose=verbose,
        )
        self.draw = True

    def recognize(self, image: np.ndarray) :
        boxes_list = face_detect(image)
        if not len(boxes_list) or not check_box(boxes_list[0]):
            return [],[]
        return self.age_gender_model.predict(image, boxes_list)

