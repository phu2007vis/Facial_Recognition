from resources.sql.sql_contronler import *
from resources.utility import SPLIT,check_box,get_time,config
import os
import glob
from recognition import Recognition
from resources.detection_model.detect import face_detect
import cv2


recog = Recognition()




permision = input("Xac nhan them nguoi moi(y/n): ")
if "n" in permision:
    print("exit insert new person")
    exit()
if "y" not in permision:
    print("y not in your anser -> exit")
    exit()
max_id,check = get_max_id()
if  check:
    print("SQL error")
    exit()
begin_index = max_id+1
with open("logfile.txt",'a+') as f:
    f.write(f"Insert new faces from new id {begin_index} ")

input_folder = os.path.join(config['data']['data_path'],"*")
for i,path_dir in enumerate(glob.glob(input_folder)):
        new_id = i+begin_index
        try:
            name = path_dir.spit("-->")[-1]
        except:
            name  = os.path.basename(path_dir).split(SPLIT)[-1]
        base_name = str(new_id) +SPLIT+name
        dir_name = os.path.dirname(path_dir)
        new_name = os.path.join(dir_name,base_name)
        os.rename(path_dir,new_name)
        print(new_name)
with open("logfile.txt",'a+') as f:
    f.write(f"to {begin_index+len(os.listdir(input_folder[:-2]))} ")
    f.write("\n")
for i,path_dir in enumerate(glob.glob(os.path.join(config['data']['data_path'],"*"))):
    name = os.path.basename(path_dir)
    id_per = name.split(SPLIT)[0]
    name = name.split(SPLIT)[-1]
    
    for image_index,image_path in enumerate(glob.glob(os.path.join(path_dir , "*"))):
        print(f"loading file: {image_path}")
        image  = cv2.imread(image_path)
     
        box  = face_detect(image)[0]
        if image_index == 0:
            time = get_time()
            insert_new_person(id_per,name,"male",2,time)
        if check_box(box):
            x1,y1,x2,y2 = box
            feature = recog.extract_feature(image,box)
            image = image[y1:y2,x1:x2,:]
            base64_image = image_to_base64(image)
        
            insert_new_feature_and_image(image_index,id_per,feature,base64_image)
        else:
            print(f"Important check file  {image_path} have no face")
            
            
        

