import torch
import torch.nn as nn


class BasicLinearModel(nn.Module):
    def __init__(self,num_class,feature_dims = [128,256],dropout = [0.3,0.5]):
        super(BasicLinearModel,self).__init__()
        self.net = []
        for i in range(len(feature_dims)-1):
            self.net.append(nn.Linear(feature_dims[i],feature_dims[i+1]))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(dropout[i]))
        self.net.append(nn.Linear(feature_dims[-1],num_class))
        self.net = nn.Sequential(*self.net)
        print(self.net)
    def forward(self,x):
        return self.net(x)
if __name__ == "__main__":
    x = torch.randn(32,128)
    y = torch.tensor(range(32), dtype=torch.int64)
    y = nn.functional.one_hot(y, num_classes=500).float()
    loss = nn.CrossEntropyLoss()
    model = BasicLinearModel(500)
    optim = torch.optim.Adam(model.parameters(),lr = 0.001)
    for i in range(200):
        optim.zero_grad()
        predict = model(x)
        l = loss(predict,y)
        l.backward()
        optim.step()
        print(l.item())
        
    import pdb;pdb.set_trace()