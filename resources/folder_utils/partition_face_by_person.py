
import os.path as osp
import os
import shutil
import pandas as pd
from tqdm import tqdm

def parition_folder(annotation_file = r"D:\download\identity_CelebA.txt",root_folder = r"D:\download\face_celeba\img_align_celeba",outfolder = r"D:\download\DataBasePartitioned"):
    proces_bar  = tqdm(os.listdir(root_folder))
    data = pd.read_csv(annotation_file,header=None,sep=" ")
    filename2id = {}
    for i in range(data.shape[0]):
        file_name,label = data.iloc[i]
        filename2id[file_name] = label

    for filename in filename2id.keys():
        proces_bar.update(1)
        destination_folder = osp.join(outfolder,str(filename2id[filename]))
        os.makedirs(destination_folder,exist_ok=True)
        file_path = osp.join(root_folder,filename)
        output_file_path = osp.join(destination_folder,filename)
        shutil.copy(file_path,output_file_path)

        
if __name__ == "__main__":
    parition_folder()