
import numpy as np
from mivolo.model.mi_volo import MiVOLO
from resources.detection_model.detect import face_detect
class Predictor:
    def __init__(self, config, verbose: bool = False):
        self.age_gender_model = MiVOLO(
            config.checkpoint,
            config.device,
            half=True,
            use_persons=config.with_persons,
            disable_faces=config.disable_faces,
            verbose=verbose,
        )
        self.draw = True

    def recognize(self, image: np.ndarray) :
        boxes_list = face_detect(image)
        return self.age_gender_model.predict(image, boxes_list)

