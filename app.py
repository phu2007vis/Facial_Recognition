from flask import Flask, render_template, Response, jsonify
import cv2
from datetime import datetime
import requests
import yaml
from utility import draw_faces,put_text
from recognition import Recognition
import numpy as np
import logging
app = Flask(__name__)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
recog = Recognition()
class Stream():
    def __init__(self):
        self.stream_type = config['video_stream']['type']
        if self.stream_type == "local":
            self.vid = cv2.VideoCapture(int(config['video_stream']['stream_uri']))
        self.image = None
    
    def init_remote(self):
        pass
    def get_frame(self):
       
        if self.stream_type == "remote":
            logging.info("remote get frame")
            # Send a GET request to the Flask server's video feed endpoint
            response = requests.get(config['video_stream']['stream_uri'], stream=True)
            if response.status_code == 200:
                # Read the raw bytes of the response
                bytes_data = bytes()
                for chunk in response.iter_content(chunk_size=1024):
                    bytes_data += chunk
                    # Check for frame boundary
                    a = bytes_data.find(b'\xff\xd8')
                    b = bytes_data.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        # Extract frame and reset bytes_data
                        frame_data = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]
                        # Decode JPEG frame
                        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                        self.image = frame
                        yield self.encode_image(frame)
                        
                    
        elif self.stream_type == "local":
            logging.info("local get frame")
            while True:
                ret, img = self.vid.read()
                if ret:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (50, 50)
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    img = cv2.putText(img, datetime.now().strftime("%H:%M:%S"), org, font,
                                    fontScale, color, thickness, cv2.LINE_AA)
                    self.image = img
                    ret, jpeg = cv2.imencode('.jpg', img)
                
                    if ret:
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    else:
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
    def encode_image(self,img):
        ret, jpeg = cv2.imencode('.jpg', img)
        if ret:
            return (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            return (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

stream = Stream()

@app.route('/detect', methods=['POST'])
def detect():
    frame = np.copy(stream.image)
    name,box = recog.recognition(frame)
    frame = draw_faces(frame,[box])
    frame = put_text(frame,name,y= 100)
    cv2.imwrite("static/result.jpg", frame)
    return jsonify({'message': 'Image saved successfully'})

@app.route("/image_feed")
def image_feed():
    return Response(stream.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False,port = 5001)
