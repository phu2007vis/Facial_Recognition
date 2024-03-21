import cv2
import requests
import numpy as np
import math
from main import draw_faces
from utility import xywh2xyxy
# URL of the Flask server streaming the video
stream_url = "http://192.168.0.100:5000/image_feed"

# Function to receive frames from the Flask server
def receive_frames():
    # Send a GET request to the Flask server's video feed endpoint
    response = requests.get(stream_url, stream=True)
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
                yield frame



class Detection:
    def __init__(self):
        caffemodel = "./resources/detection_model/Widerface-RetinaFace.caffemodel"
        deploy = "./resources/detection_model/deploy.prototxt"
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
                                   out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
        bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
        #x1,y1,w,h
        return bbox

det = Detection()
# Main function to display video stream
def main():
    for frame in receive_frames():
        box = xywh2xyxy([det.get_bbox(frame)])
        frame = draw_faces(frame,box)
        cv2.imshow("Video Stream", frame)
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
