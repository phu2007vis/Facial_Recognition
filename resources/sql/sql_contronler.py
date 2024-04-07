import yaml
import mysql.connector
import pandas as pd
import pickle
import numpy as np
import atexit
from resources.utility import image_to_base64,base64_to_image 

with open("config.yaml","r") as f:
    sql_config = yaml.safe_load(f)['sql']

enpoint  = sql_config['endpoint'].replace("haha",".rds.amazonaws.com").replace("***","chi4uk6maopa").replace("---","database-2")
user = sql_config['user']
password = sql_config['password']
database = sql_config['database']

connection = None
cursor = None

def connect_to_database():
    global connection
    connection = mysql.connector.connect(
                host=enpoint,
                user=user,
                password=password,
                database=database
    )
    global cursor
    cursor = connection.cursor()
    print("SQL connected succefuly")
    
def close_database_connection():
    global connection
    if connection:
        connection.close()

atexit.register(close_database_connection)
connect_to_database()


def insert_new_person(id:int, name:str, sex:str, age:int, time_register):
    try:
            string_command = "INSERT INTO person (id, name, sex, age, time_register) VALUES (%s, %s, %s, %s, %s);"
            cursor.execute(string_command, (id, name, sex, age, time_register))
            connection.commit()
            return [None]
    except mysql.connector.Error as e:
        print("Error while connecting to MySQL", e)
        return [e]
def get_id2name():
    try:
        string_command = "SELECT id ,name from person "
        cursor.execute(string_command)
        results = cursor.fetchall()
        if results is None:
            return [None,None]
        return [dict(results),None]
    except mysql.connector.Error as e:
        print("Error while connecting to MySQL", e)
        return [None,e]
def remove_image_and_features_by_id(id_person:int):
    try:
        string_command = "DELETE  FROM image_and_features WHERE id_person = %s;"
        cursor.execute(string_command,(id_person,))
        connection.commit()
        return [None]
    except mysql.connector.Error as e:
        print("Error while connecting to MySQL", e)
        return e
    
def remove_person_by_id(id:int):
    try:
        remove_image_and_features_by_id(id)
        string_command = "DELETE FROM person WHERE id = %s;"
        cursor.execute(string_command,(id,))
        connection.commit()
        return [None]

    except mysql.connector.Error as e:
        print("Error while connecting to MySQL", e)
        return [e]
    


def get_all_person_record():
    try:
        string_command = "SELECT * from person;"
        cursor.execute(string_command)
        results = cursor.fetchall()
        column_names = [i[0] for i in cursor.description]
        df = pd.DataFrame(results, columns=column_names)
        return [df,None]
    
    except mysql.connector.Error as e:
        print("Error while connecting to MySQL", e)
        return [None,None]
        
def insert_new_feature_and_image(index_image:int,id_person:int, features, image:str):
    features = features.reshape(-1)
    assert(features.shape[0]==128)
    features  = pickle.dumps(features)
    try:
            string_command = "INSERT INTO image_and_features (index_image, id_person, features,image) VALUES (%s, %s, %s, %s);"
            cursor.execute(string_command, (index_image, id_person,features, image))
            connection.commit()
            return [None]
    except mysql.connector.Error as e:
        print("Error while connecting to MySQL", e)
        return [e]

def get_all_features_and_labels():
    try:
        connection = mysql.connector.connect(
            host=enpoint,
            user=user,
            password=password,
            database=database
        )
        if connection.is_connected():
            cursor = connection.cursor()
            string_command = "SELECT features,id_person from image_and_features;"
            cursor.execute(string_command)
            results = cursor.fetchall()
            column_names = [i[0] for i in cursor.description]
            df = pd.DataFrame(results, columns=column_names)
            features = np.array(np.vstack((df['features'].apply(pickle.loads)).values),dtype=np.float32)
            label = df.id_person.values.tolist()
            
            cursor.close()
            connection.close()
            return [features,label,None]
    except mysql.connector.Error as e:
        print("Error while connecting to MySQL", e)
        return [None,None,e]
if __name__ == "__main__":
    import time
    print(get_all_names()[0])
    # from resources.utility import SPLIT,check_box,get_time,config
    # import os
    # import glob
    # from recognition import Recognition
    # from resources.detection_model.detect import face_detect
    # recog = Recognition()
    
    # for i,path_dir in enumerate(glob.glob(os.path.join(config['data']['data_path'],"*"))):
    #         try:
    #             int(os.base_name(path_dir.spit("-->")[0]))
    #         except:
    #             base_name = str(i) +SPLIT+os.path.basename(path_dir).split(SPLIT)[-1]
    #             dir_name = os.path.dirname(path_dir)
    #             new_name = os.path.join(dir_name,base_name)
    #             os.rename(path_dir,new_name)
                
    # for i,path_dir in enumerate(glob.glob(os.path.join(config['data']['data_path'],"*"))):
    #     name = os.path.basename(path_dir)
    #     id = name.split(SPLIT)[0]
    #     name = name.split(SPLIT)[-1]
        
    #     for image_index,image_path in enumerate(glob.glob(os.path.join(path_dir , "*"))):
    #         print(f"loading file: {image_path}")
    #         image  = cv2.imread(image_path)
    #         id_per = i
    #         box  = face_detect(image)[0]
    #         if image_index == 0:
    #             time = get_time()
    #             insert_new_person(id,name,"male",2,time)
    #         if check_box(box):
    #             x1,y1,x2,y2 = box
    #             feature = recog.extract_feature(image,box)
    #             image = image[y1:y2,x1:x2,:]
    #             base64_image = image_to_base64(image)
            
    #             insert_new_feature_and_image(image_index,id_per,feature,base64_image)
                
            

