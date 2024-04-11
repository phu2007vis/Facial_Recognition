import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data,num_class,mode = "train"):
        self.inputs ,self.targets = data
        self.targets = torch.tensor(self.targets,dtype=torch.int64)
        self.num_class = num_class
        self.mode = mode
        self.setup()
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, index):
        sample = torch.tensor(self.inputs[index]).float()+(torch.rand(128,)*0.005-0.0025)
        target = F.one_hot(self.targets[index],self.num_class).float()
        return sample, target
    def setup(self):
        my_counter = defaultdict(int)
        location_index = {}
        for i,label in enumerate(self.targets):
            label = label.item()
            my_counter[label]+=1
            if location_index.get(label) is None:
                location_index[label] = [i]
            else:
                location_index[label].append(i)
        indices = []
        for label in my_counter.keys():
            num_instance = max(1,int(0.2*my_counter[label]))
            if self.mode == "valid":
                new_indices = location_index[label][:num_instance]
            elif self.mode == "train":
                new_indices = location_index[label][num_instance:]
            assert(len(new_indices)>=1)
            indices.extend(new_indices)
        self.inputs = self.inputs[indices,:]
        self.targets = self.targets[indices]

            

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