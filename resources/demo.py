import argparse
import logging
import cv2
import torch
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging

_logger = logging.getLogger("inference")


import gdown

# Replace the URL with your Google Drive file link

def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch MiVOLO Inference")
    parser.add_argument("--output", type=str, default="output", help="folder for output results")
    parser.add_argument("--checkpoint", default=r"model_imdb_age_gender_4.22.pth", type=str, help="path to mivolo checkpoint")

    parser.add_argument(
        "--with-persons", action="store_true", default=False, help="If set model will run with persons, if available"
    )
    parser.add_argument(
        "--disable-faces", default=False, help="If set model will use only persons if available"
    )

    parser.add_argument("--device", default="cpu", type=str, help="Device (accelerator) to use.")

    return parser


def main():
    parser = get_parser()
    setup_default_logging()
    args = parser.parse_args()
    predictor = Predictor(args, verbose=True)
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
