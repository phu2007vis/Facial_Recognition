import yaml
import mysql.connector
import pandas as pd
import pickle
import numpy as np
import atexit
from resources.utility import image_to_base64,base64_to_image 

with open("config.yaml","r") as f:
    sql_config = yaml.safe_load(f)['sql']

enpoint  = "database-1.cn1puzv8rxhl.us-east-1.rds.amazonaws.com"
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
        string_command = "SELECT features,id_person from image_and_features;"
        cursor.execute(string_command)
        results = cursor.fetchall()
        column_names = [i[0] for i in cursor.description]
        df = pd.DataFrame(results, columns=column_names)
        features = np.array(np.vstack((df['features'].apply(pickle.loads)).values),dtype=np.float32)
        label = df.id_person.values.tolist()
        return [features,label,None]
    except mysql.connector.Error as e:
        print("Error while connecting to MySQL", e)
        return [None,None,e]
def get_max_id():
    try:
        string_command = "select max(id) from person"
        cursor.execute(string_command)
        results = cursor.fetchall()
        if not results:
            return [None,None]
        
        return [results[0][0],None]

    except mysql.connector.Error as e:
        print("Error while connecting to MySQL", e)
        return [e]
def check_in(id,time,time_string):
    try:
        string_command = "INSERT INTO check_in (id, time_check_in,time_string) VALUES (%s, %s , %s);"
        cursor.execute(string_command, (id, time,time_string))
        connection.commit()
        return [None]
    except mysql.connector.Error as e:
        print("Error while connecting to MySQL", e)
        return [e]
def delete_check_in_by_id(id):
    try:
        string_command = "delete from check_in where id = %s ;"
        cursor.execute(string_command, (id,))
        connection.commit()
        return [None]
    except mysql.connector.Error as e:
        print("Error while connecting to MySQL", e)
        return [e]
    
if __name__ == "__main__":
    remove_person_by_id(5)
