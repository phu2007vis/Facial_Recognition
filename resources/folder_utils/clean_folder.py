import glob
import os
import shutil


def remove_folder_inder_min_file(root_folder,min = 5):
    for sub_folder in glob.glob(os.path.join(root_folder,"*")):
        if len(os.listdir(sub_folder)) < min:
            shutil.rmtree(sub_folder)
            print(f"Remove {sub_folder} ")

if __name__ == "__main__":
    remove_folder_inder_min_file(root_folder=r"D:\download\KUB")