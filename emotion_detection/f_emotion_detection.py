import config as cfg
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
import tensorflow as tf
class predict_emotions():
    def __init__(self):
        # cargo modelo de deteccion de emociones
        self.model =  tf.compat.v1.keras.models.load_model(cfg.path_model)
    
    def preprocess_img(self,face_image,rgb=True,w=48,h=48):
        face_image = cv2.resize(face_image, (w,h))
        if rgb == False:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        face_image = face_image.astype("float") / 255.0
        face_image= img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)
        return face_image

    def get_emotion(self,img,boxes_face,thresh = 0.6):
        emotions = []
        if len(boxes_face)!=0:
            try:
                for box in boxes_face:
                    x1,y1,x2,y2 = box
                    face_image = img[y1:y2,x1:x2]
                    # preprocesar data
                    face_image = self.preprocess_img(face_image ,cfg.rgb, cfg.w, cfg.h)
                    # predecir imagen
                    prediction = self.model.predict(face_image)
                    if prediction.max(1).item()>thresh:
                        emotion = cfg.labels[prediction.argmax()]
                        emotions.append(emotion)
            except:
                import pdb;pdb.set_trace()
        else:
            emotions = []
            boxes_face = []
        return boxes_face,emotions

