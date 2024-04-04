import argparse
import logging
import cv2
import torch
from resources.mivolo.predictor import Predictor
from timm.utils import setup_default_logging

_logger = logging.getLogger("inference")






def main():


    predictor = Predictor(verbose=True)
    cam = cv2.VideoCapture(0)
    while True:
        try:
            ret,img = cam.read()
            if not ret:
                break
            with torch.no_grad():
                predictor.recognize(img)
            cv2.imshow("image",img)
            if cv2.waitKey(8) == ord("q"):
                break
        except:
            print("error")

           

if __name__ == "__main__":
    main()
