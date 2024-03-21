# -*- coding: utf-8 -*-
# @Time : 20-6-9 上午10:20
# @Author : zhuying
# @Company : Minivision
# @File : anti_spoof_predict.py
# @Software : PyCharm

import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F


from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE
}


class Detection:
    def __init__(self):
        caffemodel = "./resources/detection_model/Widerface-RetinaFace.caffemodel"
        deploy = "./resources/detection_model/deploy.prototxt"
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
                                   out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
        bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
        return bbox

class AntiSpoofPredict(Detection):
    def __init__(self,model_path):
        super(AntiSpoofPredict, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model(model_path)
        self.model.eval()
    def _load_model(self, model_path):
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input,)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        # load model weight
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        return None

    def predict(self, img):
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            result = self.model.forward(img)
            result = F.softmax(result).cpu().numpy()
        return result

import glob
from src.generate_patches import CropImage
from src.utility import xyxy2xywh
model_test = []
info_config = []
folder_path = r"D:\Silent-Face-Anti-Spoofing\resources\anti_spoof_models"
for model_path in glob.glob(os.path.join(folder_path,"*")):
    model_test.append(AntiSpoofPredict(model_path=model_path))
    h_input, w_input, model_type, scale = parse_model_name(os.path.basename(model_path))
    info_config.append(
        {
            "h_input":h_input,
            "w_input":w_input,
            "scale": scale
        }
    )
    image_cropper = CropImage()
def spoof_predict( image :np.array,image_bbox:None ):
    if image_bbox is not None:
        assert(len(image_bbox)==4)
        assert(image_bbox[0]<=image_bbox[2] and image_bbox[1] <image_bbox[3])
        
        image_bbox = xyxy2xywh(image_bbox)
    else:
        image_bbox = model_test[0].get_bbox(image)
    prediction = np.zeros((1, 3))
    for i in range(2):
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": info_config[i]['scale'],
            "out_w": info_config[i]['w_input'],
            "out_h": info_config[i]['h_input'],
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        prediction += model_test[i].predict(img)

    label = np.argmax(prediction)
    value = prediction[0][label]/len(model_test)
    # if label == 1:
    #         result_text = "RealFace Score: {:.2f}".format(value)
    #         color = (255, 0, 0)
    # else:
    #         result_text = "FakeFace Score: {:.2f}".format(value)
    #         color = (0, 0, 255)
    # cv2.rectangle(
    #         image,
    #         (image_bbox[0], image_bbox[1]),
    #         (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
    #         color, 2)
    # cv2.putText(
    #         image,
    #         result_text,
    #         (image_bbox[0], image_bbox[1] - 5),
    #         cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

    # cv2.imshow("hih",image)
    # if cv2.waitKey(0) == ord("q"):
    #     pass
    # fps = 1/test_speed
    return label,value








