from fastapi import FastAPI
import base64
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import yaml
from utility import draw_faces, put_text
from recognition import Recognition

app = FastAPI()

# Mounting static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

from typing import Optional

from pydantic import BaseModel

class imgage_base64(BaseModel):
    data: str 



with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

recog = Recognition()
def base64_to_image(base64_string):
    # Remove the data URL prefix if it exists
    if base64_string.startswith("data:image"):
        base64_string = base64_string.split(",")[1]

    # Decode the base64 string
    decoded_data = base64.b64decode(base64_string)
    
    # Convert to numpy array
    nparr = np.frombuffer(decoded_data, np.uint8)
    
    # Decode the image using cv2
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return image

@app.get("/")
def index():
   return {"message":"Home"}


@app.post('/detect')
def detect(base_64:imgage_base64):
    try:
        frame = base64_to_image(base_64.data)
        name, box = recog.recognition(frame)
        return {"Ten": f"{name}"}
    except:
        return {"Error": "Sai format"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
