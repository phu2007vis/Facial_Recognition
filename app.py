from flask import Flask, render_template, Response, jsonify
import cv2
from datetime import datetime
import yaml
from utility import draw_faces,put_text
from recognition import Recognition
import numpy as np
app = Flask(__name__)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
recog = Recognition()
class Stream():
    def __init__(self):
        self.vid = cv2.VideoCapture(config['stream_uri'])
        self.image = None

    def get_frame(self):
        if not self.vid.isOpened():
            return

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

@app.route("/")
def index():
    return render_template("index.html")

stream = Stream()

@app.route('/detect', methods=['POST'])
def detect():
    frame = np.copy(stream.image)
    name,box = recog.recognition(frame)
    frame = draw_faces(frame,[box])
    frame = put_text(frame,name)
    cv2.imwrite("static/result.jpg", frame)
    return jsonify({'message': 'Image saved successfully'})

@app.route("/image_feed")
def image_feed():
    return Response(stream.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)
