import torch
import torch.nn as nn
from resources.classification_model.BasicLinear import BasicLinearModel
from skorch.helper import predefined_split
from resources.classification_model.BasicDataLoader import CustomDataset
from resources.sql.sql_contronler import *
from skorch.callbacks import Checkpoint,EarlyStopping,EpochScoring
from resources.utility import valid_accuracy
from skorch import NeuralNetRegressor

loss_fn_map  ={
    "CrossEntropyLoss": nn.CrossEntropyLoss
}
cross_entropy_loss = nn.CrossEntropyLoss() 
class loss_fn(nn.Module):
        def forward(self,y_pred,y_target):
            return cross_entropy_loss(y_pred,y_target)
def get_predict(net,feature):
        feature = torch.tensor(feature).float()
        if len(feature.shape)==1:
            feature = torch.tensor(feature).unsqueeze(0)
            
        proba,indices = torch.softmax(torch.tensor(net.predict(feature)),dim = 1).max(1)
        return proba[0].item(),indices[0].item()
def get_classifier_model(num_class,
                            feature_dims  = [128,256],
                            dropout = [0.3,0.4],
                            lr = 0.001,
                            batch_size = 60,
                            max_epochs = 500,
                            mode = "train",
                            valid_dataset = None,
                            pretrained = None,
                            loss_fn_name = "CrossEntropyLoss",
                            device = "cpu"):
        model = BasicLinearModel(
            feature_dims=feature_dims,
            dropout=dropout,
            num_class=num_class
        )

        if mode  == "train":
            net = NeuralNetRegressor(
                module = model,
                lr = lr,
                optimizer = torch.optim.Adam,
                optimizer__weight_decay = 0.00005,
                batch_size = batch_size,
                criterion = loss_fn,
                max_epochs = max_epochs,
                train_split = predefined_split(valid_dataset),
                callbacks = [
                    EarlyStopping(patience = 13),
                    EpochScoring(valid_accuracy),
                    Checkpoint(load_best = True,dirname = "train_model")
                ]
            )
            if pretrained:
                net.initialize() 
                net.load_params(f_params = pretrained)
        else:
            net = NeuralNetRegressor(
                module = model,
                lr = lr,
                optimizer = torch.optim.Adam,
                optimizer__weight_decay = 0.00005,
                batch_size = batch_size,
                criterion = loss_fn,
                max_epochs = max_epochs,
            )
            if pretrained:
                net.initialize() 
                net.load_params(f_params = pretrained)
                net.trim_for_prediction()
            else:
                 raise EOFError
           
        return net
   

def init_and_train_model_from_scatch_pipeline(
                                              feature_dims = [128,256],
                                              dropout=  [0.3,0.4],
                                              device = "cpu",
                                              max_epochs = 50,
                                              batch_size = 64,
                                             ):
    
    num_class=get_max_id()[0]+1
    encode ,label,_ = get_all_features_and_labels()
    valid_dataset = CustomDataset((encode,label),mode = "valid",num_class=num_class)
    train_dataset = CustomDataset((encode,label),mode = "train",num_class=num_class)
    net = get_classifier_model(num_class=num_class,
                               valid_dataset=valid_dataset,
                               mode="train",
                               batch_size=batch_size,
                               max_epochs=max_epochs,
                               feature_dims=feature_dims,
                               dropout=dropout,
                               device=device
                               )
    net.fit(train_dataset,None)
if __name__ == "__main__":
    init_and_train_model_from_scatch_pipeline(max_epochs=1000)
        


            

        
        
        
            
            
            
            