
import time
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
        import pdb;pdb.set_trace
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = max(0,out[max_conf_index, 3]*width), max(out[max_conf_index, 4]*height,0), \
                                   out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
        bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
        return [bbox]

class AntiSpoofPredict(Detection):
    def __init__(self,model_path,export_model = False):
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
    def inference_torch(self) -> np.ndarray:
        y_pred = self.model(self.sample_data['sample_data_torch'])
        return y_pred.detach().cpu().numpy()
    def inference_tflite(self, tflite_model) -> np.ndarray:
        input_details = tflite_model.get_input_details()
        output_details = tflite_model.get_output_details()
        tflite_model.set_tensor(input_details[0]['index'], self.sample_data['sample_data_np'])
        tflite_model.invoke()
        y_pred = tflite_model.get_tensor(output_details[0]['index'])
        return y_pred
    def convert(self):
        self.torch2onnx()
        self.onnx2tf()
        self.tf2tflite()
        torch_output = self.inference_torch()
        tflite_output = self.inference_tflite(self.load_tflite())
        self.calc_error(torch_output, tflite_output)
    def tf2tflite(self) -> None:
        converter = tf.lite.TFLiteConverter.from_saved_model(self.tf_model_path)
        tflite_model = converter.convert()
        with open(self.tflite_model_path, 'wb') as f:
            f.write(tflite_model)
    def load_tflite(self):
        interpret = tf.lite.Interpreter(self.tflite_model_path)
        interpret.allocate_tensors()
        print(f'TFLite interpreter successfully loaded from, {self.tflite_model_path}')
        return interpret
    def load_sample_input(self,
            file_path =  None,
            target_shape: tuple = (80, 80, 3),
            normalize: bool = True
    ):
        img = cv2.resize(
                    src=cv2.imread(file_path),
                    dsize=target_shape[:2],
                    interpolation=cv2.INTER_LINEAR
                )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
        img = img.astype(np.float32)
        sample_data_np = np.transpose(img, (2, 0, 1))[np.newaxis, :, :, :]
        sample_data_torch = torch.from_numpy(sample_data_np)
        self.sample_data = {}
        self.sample_data['sample_data_np'] = sample_data_np
        self.sample_data['sample_data_torch'] = sample_data_torch
    def torch2onnx(self) -> None:
        torch.onnx.export(
            model=self.model,
            args=self.sample_data['sample_data_torch'],
            f=self.onnx_model_path,
            verbose=False,
            export_params=True,
            do_constant_folding=False,
            input_names=['input'],
            opset_version=10,
            output_names=['output'])

    def onnx2tf(self) -> None:
        onnx_model = onnx.load(self.onnx_model_path)
        onnx.checker.check_model(onnx_model)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(self.tf_model_path)
    @staticmethod
    def calc_error(result_torch, result_tflite):
        mse = ((result_torch - result_tflite) ** 2).mean(axis=None)
        mae = np.abs(result_torch - result_tflite).mean(axis=None)
        print(f'MSE (Mean-Square-Error): {mse}\tMAE (Mean-Absolute-Error): {mae}')

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
def inference_tflite( tflite_model,data) -> np.ndarray:
        input_details = tflite_model.get_input_details()
        output_details = tflite_model.get_output_details()
        tflite_model.set_tensor(input_details[0]['index'], data)
        tflite_model.invoke()
        y_pred = tflite_model.get_tensor(output_details[0]['index'])
        return y_pred
import glob
from src.generate_patches import CropImage
from src.utility import xyxy2xywh
model_test = []
info_config = []
folder_path = "./resources/anti_spoof_models"
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
        assert(len(image_bbox[0])==4)
        image_bbox = xyxy2xywh(image_bbox[0])
    else:
        image_bbox = model_test[0].get_bbox(image)[0]
    prediction = np.zeros((1, 3))
    if True:
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

    return label,value




if __name__ == "__main__":
    model =  AntiSpoofPredict(r"D:\face_liveness_detection-Anti-spoofing\resources\anti_spoof_models\2.7_80x80_MiniFASNetV2.pth",False)
    model.tflite_model_path = r"D:\face_liveness_detection-Anti-spoofing\tfmodel\spoofing.tflite"
    tf_model = model.load_tflite()
    print(inference_tflite(tf_model,np.zeros((1,3,80,80)).astype(np.float32)))
    




