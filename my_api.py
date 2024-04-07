from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import yaml
from recognition import Recognition
from resources.utility import base64_to_image
app = FastAPI()

# Mounting static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

class imgage_base64(BaseModel):
    data: str 


recog = Recognition()


@app.get("/")
def index():
   return {"message":"Home"}


@app.post('/detect')
def detect(base_64:imgage_base64):
    try:
        frame = base64_to_image(base_64.data)
        id, box = recog.recognition(frame)
        return {"Id": f"{id}"}
    except:
        return {"Error": "Sai format"}



if __name__ == "__main__":
    uvicorn.run(app)
