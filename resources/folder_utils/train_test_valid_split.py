
import os 
import shutil
from glob import glob
def read_train_valid_test_annotation(file_split):
    map_fold ={} 
    with open(file_split,'r') as f:
        data = [line.strip().split(" ") for line in f.readlines()]
    for value in data:
        file_name , fold = value
        map_fold[file_name] = fold
    return map_fold

def train_test_split_with_annotation(root_folder, annotation_file, output_folder):
    """
    Function to split files from root folder into train, validation, and test folders based on annotations.
    
    Parameters:
        root_folder (str): Path to the root folder containing files.
        annotation_file (str): Path to the annotation file.
        output_folder (str): Path to the output folder where split files will be saved.
    """
    map_fold = read_train_valid_test_annotation(annotation_file)
    map_fold_destination = {0: "train", 1: "valid", 2: "test"}

    print(f"Total number of files in root folder: {len(os.listdir(root_folder))}")

    for file_name in os.listdir(root_folder):
        fold = map_fold.get(file_name)
        destination = map_fold_destination[fold]
        input_path = os.path.join(root_folder, file_name)
        destination_folder = os.path.join(output_folder,destination)
        os.makedirs(destination_folder,exist_ok=True)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.copy(input_path, destination_path)
    
    for fold_name in os.listdir(output_folder):
        num_files = len(os.listdir(os.path.join(output_folder, fold_name)))
        print(f"{fold_name} has {num_files} files.")
if __name__ == "__main__":
    train_test_split_with_annotation(root_folder="/work/21013187/do_an_co_so_cua_phuoc/DataBase/img_align_celeba",
                                     annotation_file="/work/21013187/do_an_co_so_cua_phuoc/DataBase/train_valid_test_split.txt",
                                     output_folder="/work/21013187/do_an_co_so_cua_phuoc/DataBase/img_align_celeba_splited")
    