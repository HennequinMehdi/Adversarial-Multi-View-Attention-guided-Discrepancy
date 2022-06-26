# -*- coding: utf-8 -*-
"""
Created on Mon May  2 09:30:44 2022

@author: mehdihennequin
"""

import os

import torch 
import torch.nn as nn

import pandas as pd
import numpy as np
import random
from numpy.random import default_rng


from utils.utils import download_uci, superconduct, create_views, create_domain, zero_padding, domain
from utils.utils import kin, BaggingModels, cross_val, domain, download_uci, superconduct, create_views, zero_padding
from utils.utils_hdisc import batch_loader, split_source_target, val_split
from models.hdisc_msda import Disc_MSDANet, weighted_mse
from torch.utils.data import DataLoader, Dataset, TensorDataset

## utils preprocessing from sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


from SADISC import AMVSAD




class E2EDatasetLoader(TensorDataset):
    def __init__(self, features, targets=None):  # , transform=None
    
        self.features = features

        if targets is not None:
            self.targets = targets  # .tocsr()
        else:
            self.targets = targets

    def __len__(self):
        return self.features[0].shape[0]

    def __getitem__(self, index):
        instance = [ self.features[i][index] for i in range(len(self.features))]
        if self.targets is not None:
            target = self.targets[index]
        else:
            target = None
        return instance, target

# Set seed
seed = 1

np.random.seed(seed)
random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

X_v1, X_v2 = superconduct() # Load superconductivity dataset from UCI, 
# we receive two views
y = X_v1['critical_temp'] # stock label
std = y.std()
## we standarize the data
MinMaxscaler = MinMaxScaler()
scaler = StandardScaler()
scaler2 = StandardScaler(with_mean=False)

y = MinMaxscaler.fit_transform(scaler.fit_transform(y.to_numpy().reshape(-1,1)))


##encode the features in numeric
encodeur = preprocessing.LabelEncoder()
X_v2.iloc[:,87] = encodeur.fit_transform(X_v2.iloc[:,87])



X_s = []
y_D = []
X_D, cuts = create_domain(X_v1) ## we create different domains 


X_v1 = X_v1.drop([X_v1.columns[X_v1.columns.get_loc('critical_temp')]],1)## remove label
X_v2 = X_v2.drop([X_v2.columns[X_v2.columns.get_loc('critical_temp')]],1)## remove label


index = X_v1.index
columns= X_v1.columns
X_v1 = MinMaxscaler.fit_transform(scaler.fit_transform(X_v1))
X_v1 = pd.DataFrame(data =X_v1,index=index,columns = columns )

index = X_v2.index
columns= X_v2.columns
X_v2 =  MinMaxscaler.fit_transform(scaler2.fit_transform(X_v2))
X_v2 =  pd.DataFrame(data =X_v2,index=index,columns = columns )

rng = default_rng()
SelectFeatures = rng.choice((X_v1.shape[1]-1), size=int((X_v1.shape[1]/2)), replace=False) ## we select randomly features
dataframe_views = list(pd.concat([X_v1,X_v2], axis=1).sample(n=int((X_v1.shape[1] + X_v2.shape[1])/2),axis='columns').columns)
dataframe_views1 = []
dataframe_views2 = []

for i in dataframe_views :
    if i in list(X_v1.columns):
        dataframe_views1.append(i)
    else:
        dataframe_views2.append(i)

    
for i in range(len(cuts)):
    X_v = []
    y_D.append(y[cuts[i],:])
    X_v.append(X_v1.iloc[cuts[i],:])
    X_v.append(X_v2.iloc[cuts[i],:])
    X_s.append(X_v)
    
    
name_columns = [X_s[0][0].copy().drop(dataframe_views1, axis=1).columns, X_s[0][1].copy().drop(dataframe_views2, axis=1).columns, pd.concat([X_s[0][0].copy()[dataframe_views1],X_s[0][1].copy()[dataframe_views2]],1).columns]    
    
def get_feature_extractor():
    return nn.ModuleList([
            nn.Linear(84,50), nn.LeakyReLU(),nn.Dropout(0.1),
            # nn.Linear(256,50), nn.LeakyReLU(),
            nn.Linear(50, 25), nn.Tanh()])

def get_predictor(output_dim=1):
    return nn.ModuleList([
            nn.Linear(25,9),nn.ReLU(),
            # nn.Linear(1024, 500, bias=False), 
            # nn.Linear(50,25, bias=False), nn.LeakyReLU(), 
            # nn.Linear(10,5, bias=False), nn.LeakyReLU(),
            nn.Linear(9, output_dim)])


def get_predictor2(output_dim=1):
    return nn.ModuleList([
            nn.Linear(25,9),nn.ReLU(),
            # nn.Linear(50,25, bias=False), nn.LeakyReLU(), 
            # nn.Linear(10,5, bias=False), nn.LeakyReLU(),
            nn.Linear(9, output_dim)])

def get_discriminator(output_dim=1):
    return nn.ModuleList([
            nn.utils.spectral_norm(nn.Linear(25,9),eps=1e-3),nn.ReLU(),
            # nn.Linear(50,25, bias=False), nn.LeakyReLU(), 
            # nn.Linear(10,5, bias=False), nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Linear(9, output_dim),eps=1e-3)])








def get_discriminator_A(output_dim=1):
    return nn.ModuleList([
            nn.Linear(25,9),nn.ReLU(),
            # nn.Linear(50,25, bias=False), nn.LeakyReLU(), 
            # nn.Linear(10,5, bias=False), nn.LeakyReLU(),
            nn.Linear(9, output_dim)])




input_shape = [84,84]
input_shape_target = 84
hidden_layer_size_attention = 256
learning_rate = 0.0001
lr = 0.001
epochs_adapt = 100
batch_size = 256
epochs_h_disc, epochs_feat, epochs_alpha, epochs_pred = 1, 1, 1, 1

list_modelViews = nn.ModuleList([get_predictor(output_dim=1),get_predictor2(output_dim=1)])
list_modelDiscriminator = nn.ModuleList([get_discriminator(output_dim=1),get_discriminator(output_dim=1)])
params= {'device' : device, 'n_views': 2, 'num_heads' : 2,'loss': torch.nn.MSELoss(), 'input_shape':input_shape, 'input_shape_target':input_shape_target, 'hidden_layer_size_attention':hidden_layer_size_attention,
         'learning_rate':learning_rate,'min_pred': -np.inf, 'max_pred': np.inf}
params['feature_extractor'] = get_feature_extractor()
params['h_v'] = list_modelViews
params['discriminator'] = get_discriminator(output_dim=1)
params['columns'] = name_columns




paramsAHD = {'input_dim': 84, 'output_dim': 1, 'n_sources': 2, 'loss': torch.nn.MSELoss() ,'weighted_loss': weighted_mse, 'min_pred': -np.inf, 'max_pred': np.inf}
paramsAHD['feature_extractor'] = get_feature_extractor()
paramsAHD['h_pred'] = get_predictor(output_dim=1)
paramsAHD['h_disc'] = get_discriminator_A()

                                    
Score_target = {}
Score_source = {}
err_t = 0
err_s = 0
for source in [0, 1, 2, 3]:
    for target in [0, 1, 2, 3]:
        if source != target:
            print("------------------------------------------------------------Source Doamin:",source)
            print("------------------------------------------------------------Target Doamin:",target)
                
            params['experiments'] = [str(source),str(target)]
            for method in ["AMVSAD","AHD-MSDA"]:
                    
                if not method in Score_target:
                    Score_target[method] = []
                    Score_source[method] = []
                
                X_sv, X_unlabeled, y_s, y_unlabeled = val_split([X_s[source][0].drop(dataframe_views1, axis=1).to_numpy(),X_s[source][1].drop(dataframe_views2, axis=1).to_numpy()], [y_D[source],y_D[source]], frac_train = 0.75)
                X_t =  torch.from_numpy(np.concatenate([X_s[target][0][dataframe_views1].to_numpy(),X_s[target][1][dataframe_views2].to_numpy()],1)).type(torch.cuda.FloatTensor).to(device)
                y_t = torch.from_numpy(y_D[target]).type(torch.cuda.FloatTensor).to(device)
                
                X_v = [torch.from_numpy(np.concatenate((X_sv[0],np.ones((X_sv[0].shape[0],(X_t.shape[1]-X_sv[0].shape[1])))),1)).type(torch.cuda.FloatTensor).to(device),
                       torch.from_numpy(np.concatenate((X_sv[1],np.ones((X_sv[1].shape[0],(X_t.shape[1]-X_sv[1].shape[1])))),1)).type(torch.cuda.FloatTensor).to(device)]
            
                X_v_unlabeled = [torch.from_numpy(np.concatenate((X_unlabeled[0],np.ones((X_unlabeled[0].shape[0],(X_t.shape[1]-X_unlabeled[0].shape[1])))),1)).type(torch.cuda.FloatTensor).to(device),
                                 torch.from_numpy(np.concatenate((X_unlabeled[1],np.ones((X_unlabeled[1].shape[0],(X_t.shape[1]-X_unlabeled[1].shape[1])))),1)).type(torch.cuda.FloatTensor).to(device)]
                y_v = torch.from_numpy(y_s[0]).type(torch.cuda.FloatTensor).to(device)
                y_unlabeled = torch.from_numpy(y_unlabeled[0]).type(torch.cuda.FloatTensor).to(device)
                
                if method == "AMVSAD":
                    # print("uncoment for AMVSAD")
                    
                    model = AMVSAD(params).to(device)
                    err_s, err_t  = model.fit(X_v,X_t,y_v, y_t, X_v_unlabeled, y_unlabeled, stopping_crit =5,num_epochs=epochs_adapt, batch_size = batch_size)
                    
                    
                    
                elif method == "AHD-MSDA":
                    # print("uncoment for AHD-MSDA")

                    model = Disc_MSDANet(paramsAHD).to(device)
                    opt_feat = torch.optim.Adam([{'params': model.feature_extractor.parameters()}],lr=lr)
                    opt_pred = torch.optim.Adam([{'params': model.h_pred.parameters()}],lr=lr)
                    opt_disc = torch.optim.Adam([{'params': model.h_disc.parameters()}],lr=lr)
                    opt_alpha = torch.optim.Adam([{'params': model.alpha}],lr=lr)
                    model.optimizers(opt_feat, opt_pred, opt_disc, opt_alpha)
                    for epoch in range(epochs_adapt):
                        model.train()
                        train_dataset = E2EDatasetLoader(X_v, y_v)
                        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        for i, (x_bs, y_bs) in enumerate(dataloader):   
                            y_bs = [y_bs,y_bs]
                            ridx = np.random.choice(X_t.shape[0], batch_size)
                            x_bt = X_t[ridx,:]
                            #Train h to minimize source loss
                            for e in range(epochs_pred):
                                model.train_prediction(x_bs, x_bt, y_bs, pred_only=False)

                    #Train h' to maximize discrepancy
                            for e in range(epochs_h_disc):
                                model.train_h_discrepancy(x_bs, x_bt, y_bs)
                
                    #Train phi to minimize discrepancy
                            for e in range(epochs_feat):
                                model.train_feat_discrepancy(x_bs, x_bt, y_bs, mu=0)
                    
                    #Train alpha to minimize discrepancy
                            for e in range(epochs_alpha):
                                model.train_alpha_discrepancy(x_bs, x_bt, y_bs, clip=1, lam_alpha=0.1)

                        model.eval()
                        print(model.alpha)
                        err_s, disc = model.compute_loss(X_v_unlabeled, X_t, [y_unlabeled,y_unlabeled])
                        err_t = model.loss(y_t, model.predict(X_t))
                        print('Epoch: %i/%i (h_pred); Train loss: %.3f ; Disc: %.3f ; Test loss: %.3f'%(epoch+1, epochs_adapt, err_s.item(), disc.item(), err_t.item()))
                        
                
                Score_target[method].append(err_t.item())
                Score_source[method].append(err_s.item())
                
                        
                        
            pd.DataFrame(Score_target).to_csv("./dataset/results/Superconductivity_target_our_method_and_AHD-MSDA"+".csv")
            pd.DataFrame(Score_source).to_csv("./dataset/results/Superconductivity_source_our_method_and_AHD-MSDA"+".csv")