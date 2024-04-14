import os

def count_file(root_folder):
    sum = 0 
    for id in os.listdir(root_folder):
        input_folder = os.listdir(os.path.join(root_folder,id))
        sum+= len(input_folder)
    return sum