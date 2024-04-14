
import os
import shutil
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

def copy_file(args):
    root_folder, filename, label, outfolder, skip_empty = args
    destination_folder = os.path.join(outfolder, str(label))
    os.makedirs(destination_folder, exist_ok=True)
    file_path = os.path.join(root_folder, filename)
    if not os.path.exists(file_path):
        if skip_empty:
            return
    output_file_path = os.path.join(destination_folder, filename)
    shutil.copy(file_path, output_file_path)

def partition_folder(annotation_file=r"D:\download\identity_CelebA.txt", root_folder=r"D:\download\face_celeba\img_align_celeba", outfolder=r"D:\download\DataBasePartitioned",skip_empty = False):
    data = pd.read_csv(annotation_file, header=None, sep=" ")
    filename2id = {file_name: label for file_name, label in zip(data[0], data[1])}

    args_list = [(root_folder, filename, label, outfolder,skip_empty) for filename, label in filename2id.items()]

    with Pool() as pool:
        list(tqdm(pool.imap(copy_file, args_list), total=len(args_list), desc="Processing"))


        
if __name__ == "__main__":
    partition_folder()