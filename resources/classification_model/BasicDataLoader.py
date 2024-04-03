import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data,num_class):
        self.inputs ,self.targets = data
        self.targets = torch.tensor(self.targets,dtype=torch.int64)
        self.num_class = num_class
    
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, index):
        sample = torch.tensor(self.inputs[index]).float()
        target = F.one_hot(self.targets[index],self.num_class).float()
        return sample, target


def get_dataloader(data,num_class,batch_size= 32,shuffle = True,num_workers = 4):
    custom_dataset = CustomDataset(data,num_class)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


if __name__ == "__main__":
    import numpy as np
    inputs = np.random.rand(6,128)
    target = np.arange(6)
    dataset = CustomDataset((inputs,target),120)
    import pdb;pdb.set_trace()