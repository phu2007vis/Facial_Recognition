import yaml
import mysql.connector
import pandas as pd
import pickle
import numpy as np
with open("config.yaml","r") as f:
    config = yaml.safe_load(f)['sql']

enpoint  = config['endpoint'].replace("haha",".rds.amazonaws.com").replace("***","chi4uk6maopa").replace("---","database-2")
user = config['user']
password = config['password']
database = config['database']

def insert_new_person(id:int, name:str, sex:str, age:int, time_register,conection = None):
    try:
        shutdown = False
        if connection is None:
            shutdown = True
            connection = mysql.connector.connect(
                host=enpoint,
                user=user,
                password=password,
                database=database
            )
        if connection.is_connected():
            cursor = connection.cursor()
            string_command = "INSERT INTO person (id, name, sex, age, time_register) VALUES (%s, %s, %s, %s, %s);"
            cursor.execute(string_command, (id, name, sex, age, time_register))
            connection.commit()
            print("Record inserted successfully")

    except mysql.connector.Error as e:
        print("Error while connecting to MySQL", e)
        if connection.is_connected() and shutdown:
            cursor.close()
            connection.close()
        return e
    
    finally:
        if connection.is_connected() and shutdown:
            cursor.close()
            connection.close()
def remove_person_by_id(id:int,conection = None):

    try:
        shutdown = False
        if conection is None:
            shutdown  = True
        connection = mysql.connector.connect(
            host=enpoint,
            user=user,
            password=password,
            database=database
        )
        if connection.is_connected():
            cursor = connection.cursor()
            string_command = "DELETE FROM person WHERE id = %s;"
            cursor.execute(string_command,(id,))
            connection.commit()
            cursor.close()
            connection.close()

    except mysql.connector.Error as e:
        print("Error while connecting to MySQL", e)
           
        if connection.is_connected():
            cursor.close()
            connection.close()
        return e
def remove_image_and_features_by_id(id_person:int):
    try:
        connection = mysql.connector.connect(
            host=enpoint,
            user=user,
            password=password,
            database=database
        )
        if connection.is_connected():
            cursor = connection.cursor()
            string_command = "DELETE  FROM image_and_features WHERE id_person = %s;"
            cursor.execute(string_command,(id_person,))
            connection.commit()
            cursor.close()
            connection.close()

    except mysql.connector.Error as e:
        print("Error while connecting to MySQL", e)
           
        if connection.is_connected():
            cursor.close()
            connection.close()
        return e


def get_all_person_record():
    try:
        connection = mysql.connector.connect(
            host=enpoint,
            user=user,
            password=password,
            database=database
        )
        if connection.is_connected():
            cursor = connection.cursor()
            string_command = "SELECT * from person;"
            cursor.execute(string_command)
            results = cursor.fetchall()
            column_names = [i[0] for i in cursor.description]
            df = pd.DataFrame(results, columns=column_names)
            cursor.close()
            connection.close()
            return df

    except mysql.connector.Error as e:
        print("Error while connecting to MySQL", e)
           
        if connection.is_connected():
            cursor.close()
            connection.close()
def insert_new_feature_and_image(index_image:int,id_person:int, features, image:str):
    features = features.reshape(-1)
    assert(features.shape[0]==128)
    features  = pickle.dumps(features)
    try:
        connection = mysql.connector.connect(
            host=enpoint,
            user=user,
            password=password,
            database=database
        )
        if connection.is_connected():
            cursor = connection.cursor()
            string_command = "INSERT INTO image_and_features (index_image, id_person, features,image) VALUES (%s, %s, %s, %s);"
            cursor.execute(string_command, (index_image, id_person,features, image))
            connection.commit()
            print("Record inserted successfully")

    except mysql.connector.Error as e:
        print("Error while connecting to MySQL", e)
        if connection.is_connected():
            cursor.close()
            connection.close()
        return e
    
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

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
            return features,label

    except mysql.connector.Error as e:
        print("Error while connecting to MySQL", e)
           
        if connection.is_connected():
            cursor.close()
            connection.close()
insert_new_person(2,"phuoc","male",30,"2022-12-11")
arr = np.arange(-1,127)


insert_new_feature_and_image(3,2,arr,"hi")
remove_image_and_features_by_id(0)
import pdb;pdb.set_trace()

