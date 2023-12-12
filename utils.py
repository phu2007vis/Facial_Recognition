import base64
import  cv2
import pickle
import face_recognition
import numpy
import numpy as np

import os 
import glob

class FacialVertification:
    def __init__(self,encode_data_path = "data.pkl",path_data_base  = "DataBase"):

        self.encode_data_path = encode_data_path
        self.path_data_base = path_data_base
        self.names,self.encodes = self._load_data_face(encode_data_path)

    def _load_data_face(self,face_data_path ):

        with open(face_data_path, "rb") as f:
            names = pickle.load(f)
            encodes = pickle.load(f)
        return names,encodes

    def gen_data_encode(self):
        

        self.names = []
        self.encodes = []
        print(os.getcwd())
        print(glob.glob(os.path.join(self.path_data_base,"*")))
        for path_dir in glob.glob(os.path.join(self.path_data_base,"*")):
            
            name = path_dir.split("\\")[1]
        
            self.names += [name] * len(glob.glob(os.path.join(path_dir , "*")))
        
            for image_path in glob.glob(os.path.join(path_dir , "*")):
                print(image_path)
                image = face_recognition.load_image_file(image_path)
                face_encoding = face_recognition.face_encodings(image)[0]
                self.encodes.append(face_encoding)
       

    def add_new_image(self,frame,name : str):
        locations = face_recognition.face_locations(frame)
        if len(locations)==1:
            face_encoding = face_recognition.face_encodings(frame,known_face_locations=locations,num_jitters = 2)[0]
            self.encodes.append(face_encoding)
            self.names.append(name)
            return True
        else:
            return False


    def save_data_pkl(self,file_path=None):
        if file_path == None:
            file_path= self.encode_data_path

        with open(file_path, "wb") as f:
            pickle.dump(self.names, f)
            pickle.dump(self.encodes, f)
    def _get_face_encodes(self,frame):
        locations = face_recognition.face_locations(frame)
        face_encodes = face_recognition.face_encodings(frame, locations)
        return locations,face_encodes
    def draw_face(self,frame,location,name):

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 2

        y1,x2,y2,x1 = location
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        frame = cv2.putText(frame, name, (x1, y1), font, fontScale, color, thickness, cv2.LINE_AA)
        return frame

    def check_face(self,frame,draw_image = False,threshold_value = 0.45):
        names = []
        locations,face_encodes = self._get_face_encodes(frame)

        for location, face_encode in zip(locations, face_encodes):
            face_distances = face_recognition.face_distance(self.encodes, face_encode)
            index = numpy.argmin(face_distances)
            value = face_distances[index]
            name = "unknow"
            if value < threshold_value:
                name = self.names[index]
            names.append(name)
            if draw_image:
                location = [int(x) for x in location]
                frame = self.draw_face(frame,location,name)
        return names,frame

def decode_image(encoded_image_string):
    # Decode base64 string to bytes
    decoded_image = base64.b64decode(encoded_image_string)

    # Convert bytes to NumPy array
    numpy_image = np.frombuffer(decoded_image, dtype=np.uint8)

    # Decode NumPy array to a cv2 image
    cv2_image = cv2.imdecode(numpy_image, cv2.IMREAD_COLOR)

    return cv2_image
   
def encode_numpy_to_base64(numpy_array):
    jpg_img = cv2.imencode('.jpg', numpy_array)
    b64_string = base64.b64encode(jpg_img[1]).decode('utf-8')
    return b64_string

# if __name__ == "__main__":
    
    # face_vertificator = FacialVertification()
    # count = 0
    # while True:
    #         # count += 1
    #         # if count % 4 != 0:
    #         #     continue
    #         ret,frame = cam.read()


            # if cv2.waitKey(2) == ord('q'):
            #     break
            # if cv2.waitKey(2) == ord('v'):
            #     names, frame = face_vertificator.check_face(frame,True)
            #     cv2.imshow("frame", frame)
            #     cv2.waitKey(3000)
            #     continue
            # cv2.imshow("frame",frame)
            # if cv2.waitKey(2) == ord('s'):
            #     name = input("nhap ten")
            #     if face_vertificator.add_new_image(frame,name):
            #         print("add succse")
            #     else:
            #         print("can succed")
# image = cv2.imread(r"C:\Users\phuoc\OneDrive\Pictures\quang_cao.jpg")
# import base64
# import numpy as np



# image = cv2.imread(r"C:\Users\phuoc\OneDrive\Pictures\quang_cao.jpg")
# encoded_image = encode_numpy_to_base64(image)
# decoded_image = decode_image(encoded_image)
# cv2.imshow("image",decoded_image)
# cv2.waitKey(0)



   