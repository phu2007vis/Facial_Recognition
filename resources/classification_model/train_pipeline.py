import torch
import torch.nn as nn
import torch.nn.functional as F
from resources.classification_model.BasicDataLoader import get_dataloader
from resources.classification_model.BasicLinear import BasicLinearModel
from sklearn.model_selection import train_test_split


loss_fn_map  ={
    "CrossEntropyLoss": nn.CrossEntropyLoss
}
def train_model_one_epoch(model, device, dataloader, loss_fn, optim, epoch, column_size=64):
    model.train().to(device)
    cur_column = 0
    for i, (x, target) in enumerate(dataloader):
        cur_column += x.shape[0]
        x = x.to(device)
        target = target.to(device)
        predict = model(x)
        loss = loss_fn(predict, target)
        loss.backward()
        print(f"Current loss at iter {i+1} of epoch {epoch}: {loss.item()}")
        if cur_column >= column_size:
            optim.step()
            optim.zero_grad()
            cur_column = 0  
    return model
def init_and_train_model_from_scatch_pipeline(data,
                                              num_class,
                                              feature_dims,
                                              dropout,
                                              device,
                                              max_epochs,
                                              batch_size,
                                              test_size = 0.2,
                                              loss_type = "CrossEntropyLoss" ,
                                              caculumn_size = 64):
    x,y = data
    X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=test_size)
    
    
    model = BasicLinearModel(num_class=num_class,feature_dims=feature_dims,dropout=dropout)
    optim = torch.optim.Adam(model.parameters(),lr = 0.001,weight_decay= 0.0000001)
    train_dataloader = get_dataloader(num_class = num_class,data=(X_train,Y_train),batch_size=batch_size)
    test_dataloader = get_dataloader(num_class=num_class,data=(X_test,Y_test),batch_size=batch_size)
    loss_fn =  loss_fn_map[loss_type]()
    for i in range(max_epochs):
        # model = train_model_one_epoch(model = model,
        #                        device = device,
        #                        dataloader=train_dataloader,
        #                        loss_fn=loss,
        #                        epoch=i,
        #                        column_size=caculumn_size,
        #                        optim=optim)
        cur_column = 0 
        for iter, (x, target) in enumerate(train_dataloader):
            cur_column += x.shape[0]
            x = x.to(device)
            target = target.to(device)
            predict = model(x)
            loss = loss_fn(predict, target)
            loss.backward()

            print(f"Current loss at iter {iter+1} of epoch {i+1}: {loss.item()}")
            if cur_column >= caculumn_size:
                optim.step()
                optim.zero_grad()
                cur_column = 0  
        
    
    
    
        
        
        
        