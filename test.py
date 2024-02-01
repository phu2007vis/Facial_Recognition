from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN()

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='casia-webface').eval()

from PIL import Image

img = Image.open(r"C:\Users\phuoc\OneDrive\Pictures\th.jpg")

# Get cropped and prewhitened image tensor
img_cropped = mtcnn(img)
import time
t = time.time()
with torch.no_grad():
# Calculate embedding (unsqueeze to add batch dimension)
    img_embedding = resnet(img_cropped.unsqueeze(0))
print(time.time()-t)

import pdb;pdb.set_trace()