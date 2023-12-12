
import uvicorn
from pydantic import BaseModel
from utils import *
import cv2

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


class Data(BaseModel):
    image: str
    return_frame: str = "y"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
verticator = FacialVertification()
verticator.gen_data_encode()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "message": "Hello, World!"})

@app.post("/vertification")
def vertification(data:Data):

    image = decode_image(data.image)
    image_resized = cv2.resize(image,(300,300))
    data = {}
    names, frame = verticator.check_face(image_resized,True)
    image_string =  encode_numpy_to_base64(frame)
    data['names'] = names
    data['image'] = image_string
    return JSONResponse(content=data)



if __name__ == "__main__":
    uvicorn.run("app:app","--host","192.168.0.107","--port","80")
