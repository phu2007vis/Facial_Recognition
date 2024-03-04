from facenet_pytorch import InceptionResnetV1
import math
import torch
import os
import onnx
from onnx_tf.backend import prepare
import cv2
import numpy as np
import tensorflow as tf
resnet = InceptionResnetV1(pretrained='casia-webface').eval()
tmpdir = "tfmodel_embed"
os.makedirs(tmpdir,exist_ok=True)
tf_model_path = os.path.join(tmpdir, 'tf_model')
onnx_model_path = os.path.join(tmpdir, 'model.onnx')
tflite_model_path  = os.path.join(tmpdir,"embeding_model.tflite")
def process_image(image_path):
    image = np.asarray(cv2.resize(cv2.imread(image_path),(160,160)),dtype=np.float32)
    rgb = (cv2.cvtColor(image,cv2.COLOR_BGR2RGB)-127.5)/128
    rgb = torch.from_numpy(np.transpose(np.expand_dims(rgb,axis=0),(0,3,1,2)))
    return rgb
input_data = process_image(r"D:\image\phuoc3.png")
input_data2 = process_image(r"D:\image\phuoc5.png")
input_data3 = process_image(r"D:\image\phuoc4.png")
def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
def convert():
    torch.onnx.export(
                model=resnet,
                args=input_data,
                f=onnx_model_path,
                verbose=False,
                export_params=True,
                do_constant_folding=False,
                input_names=['input'],
                opset_version=10,
                output_names=['output'])
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_model_path)
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    tflite_model = converter.convert()
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

def cosine_similarity(face_encodings, face_to_compare):
    """
    Calculate cosine similarity between a face encoding and a list of face encodings.

    :param face_encodings: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the cosine similarity for each face in the same order as the 'face_encodings' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    # Dot product between face_encodings and face_to_compare
    dot_product = np.sum(face_encodings * face_to_compare, axis=1)

    # Magnitudes of the vectors
    magnitudes = np.sqrt(np.sum(face_encodings**2, axis=1)) * np.sqrt(np.sum(face_to_compare**2))

    # Avoid division by zero
    magnitudes[magnitudes == 0] = np.finfo(float).eps

    # Cosine similarity
    similarity = dot_product / magnitudes

    return similarity
# convert()

# output_pytorch = resnet(input_data)
tflite_model = tf.lite.Interpreter(tflite_model_path)
tflite_model.allocate_tensors()

input_details = tflite_model.get_input_details()
output_details = tflite_model.get_output_details()
tflite_model.set_tensor(input_details[0]['index'],input_data.numpy())
tflite_model.invoke()
output_tflite  =  tflite_model.get_tensor(output_details[0]['index'])
tflite_model.set_tensor(input_details[0]['index'],input_data2.numpy())
tflite_model.invoke()
output_tflite2  =  tflite_model.get_tensor(output_details[0]['index'])

tflite_model.set_tensor(input_details[0]['index'],input_data3.numpy())
tflite_model.invoke()
output_tflite3  =  tflite_model.get_tensor(output_details[0]['index'])

import pdb;pdb.set_trace()
