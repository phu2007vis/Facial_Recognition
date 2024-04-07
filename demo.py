import argparse
import logging
import cv2
import torch
from resources.mivolo.predictor import Predictor
from timm.utils import setup_default_logging
from resources.utility import put_text
_logger = logging.getLogger("inference")


def main():
    predictor = Predictor(verbose=True)
    cam = cv2.VideoCapture(0)
    while True:
        
            ret,img = cam.read()
            if not ret:
                break
            with torch.no_grad():
                try:
                    ages,gioi_tinhs = predictor.recognize(img)
                    if  len(ages):
                        age = ages[0]
                        put_text(img,f"Age : {age}")
                    if len(gioi_tinhs):
                        gioi_tinh = gioi_tinhs[0]
                        put_text(img,f"Sex : {gioi_tinh}",(50))
                except:
                    pass
            cv2.imshow("image",img)
            if cv2.waitKey(8) == ord("q"):
                break


if __name__ == "__main__":
    main()
