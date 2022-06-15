# -*- coding: utf-8 -*-
"""
"""

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler


import torch
import torch.nn as nn
import torch.nn.functional as F





import torch
# from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset, TensorDataset
import logging
import numpy as np
from scipy import sparse

torch.manual_seed(1)
np.random.seed(1)

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)



class E2EDatasetLoader(TensorDataset):
    def __init__(self, features, targets=None):  # , transform=None
    
        self.features = features

        if targets is not None:
            self.targets = targets  
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

class SANNetwork(nn.Module):
    def __init__(self, input_size, hidden_layer_size = 100, num_heads=2, device="cuda"):
        super(SANNetwork, self).__init__()
        self.device  = device
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=0)
        self.num_heads = num_heads
        self.multi_head = nn.ModuleList([nn.Linear(input_size, input_size) for k in range(num_heads)])
        self.q = nn.ModuleList([nn.Linear(input_size, input_size, bias=False) for k in range(num_heads)])
        
    def forward_attention(self, input_space, return_softmax=True):
        placeholder = torch.zeros(input_space.shape).to(self.device)
        for k in range(len(self.multi_head)):
            attended_matrix = self.multi_head[k](input_space)
            attended_matrix = self.q[k](attended_matrix)
            if return_softmax:
                attended_matrix = self.activation(attended_matrix)
                attended_matrix = self.softmax(attended_matrix)
            placeholder = torch.add(placeholder,attended_matrix)
        placeholder /= len(self.multi_head)
        out = placeholder
        return out


    def forward(self, x):

        # attend and aggregate
        out = self.forward_attention(x)
        # out = self.sigmoid(self.activation(self.fc1(out)))
        # print(len(out))
        # out = self.softmax(out)
        


        return out

    def get_attention(self, x):
        return self.forward_attention(x, return_softmax=True)

    def get_softmax_hadamand_layer(self):
        return self.get_mean_attention_weights()

    
class AMVSAD(nn.Module):
    
    def __init__(self, params):
        super(AMVSAD, self).__init__()
        self.device = params['device']
        self.n_views = params['n_views']
        self.num_heads = params['num_heads']
        # Parameters of hidden, fully-connected layers, feature learning component.
        self.feature_extractor = params['feature_extractor']
        # Parameter of the final regressor.
        self.h_v = params['h_v']
        self.min_pred = params['min_pred']
        self.max_pred = params['max_pred']
        self.h_disc = params['discriminator'] 
        self.loss = params['loss']
        self.learning_rate = params['learning_rate']
        self.model_attention =  nn.ModuleList([SANNetwork(params['input_shape'][k], hidden_layer_size = params['hidden_layer_size_attention'], num_heads = self.num_heads, device=self.device).to(self.device) for k in range(self.n_views)])
        self.model_alpha_attention_layer = SANNetwork(self.n_views, hidden_layer_size = params['hidden_layer_size_attention'], num_heads = self.num_heads, device=self.device).to(self.device)
        self.model_attention_target = SANNetwork(params['input_shape_target'], hidden_layer_size = params['hidden_layer_size_attention'], num_heads = self.num_heads, device=self.device).to(self.device)
        self.alpha = torch.nn.Parameter(torch.Tensor(np.ones(self.n_views)/self.n_views), requires_grad=True) 
        # self.alpha = torch.Tensor(np.ones(self.n_views)/self.n_views) 
        #Paramete
        


    def forward_features(self, X_s, X_t):
        
        # # Attention layer
        sx, tx = X_s.copy(), X_t.clone()
        for i in range(self.n_views):
            sx[i] = self.model_attention[i](sx[i])*sx[i] + sx[i]
            

        #Feature extractor
            for hidden in self.feature_extractor:
                sx[i] = hidden(sx[i])
            
        
        tx = self.model_attention_target(tx)*tx + tx
        for hidden in self.feature_extractor:
            tx = hidden(tx)
        return sx, tx
    
    def forward(self, X_s, X_t):
        """
        Forward pass

        """
        
        sx, tx = self.forward_features(X_s,X_t)
        # Feature extractor

            
        # Predictor h
        y_spred = []
        for i in range(self.n_views):
            y_sx = sx[i].clone()
            for hidden in self.h_v[i]:
                y_sx = hidden(y_sx)
            y_spred.append(self.clamp(y_sx))
    
        y_tpred = []
        for i in range(self.n_views):    
            y_tx = tx.clone()
            for hidden in self.h_v[i]:
                y_tx = hidden(y_tx)
            y_tpred.append(self.clamp(y_tx))
            
        # Discrepant h'
        y_sdisc = []
        for i in range(self.n_views):
            y_tmp = sx[i].clone()
            for hidden in self.h_disc:
                y_tmp = hidden(y_tmp)
            y_sdisc.append(self.clamp(y_tmp))
            
            
        y_tmp = tx.clone()
        for hidden in self.h_disc:
            y_tmp = hidden(y_tmp)
        y_tdisc = self.clamp(y_tmp)
        
        return y_spred, y_sdisc, y_tpred, y_tdisc
    

            
            
    def train_h(self, X_s, X_t, y_s, clip = 1):
        
        #Training
        self.train()
        
        #Prediction training
        parameters = nn.ModuleList([self.h_v, self.feature_extractor, self.model_attention]).parameters()
        y_spred, y_sdisc, y_tpred, y_tdisc = self.forward(X_s, X_t)
        loss_pred = self.weighted_loss(y_spred,y_s) 
        self.optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        self.optimizer.zero_grad()
        loss_pred.backward(retain_graph=True)
        
        #Optimization step
        torch.nn.utils.clip_grad_norm_(self.parameters(),clip)

        self.optimizer.step()
        

        return loss_pred
    

    
    
    def train_h_discrepancy(self, X_s, X_t, y_s,clip = 1):
        """
        Train h to maximize the discrepancy

        """
        #Training
        self.train()
        self.feature_extractor.eval()
        #Discrepancy training
        parameters = self.h_disc.parameters()
        y_spred, y_sdisc, y_tpred, y_tdisc = self.forward(X_s, X_t)
        disc =  -1*torch.abs(self.weighted_loss(y_spred, y_sdisc)-self.target_loss_(y_tpred, y_tdisc))

        loss_disc = disc
        self.optimizer = torch.optim.Adam(parameters, lr= self.learning_rate)
        self.optimizer.zero_grad()
        loss_disc.backward(retain_graph=True)
        
        #Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(),clip)

        #Optimization step

        self.optimizer.step()
        
        return disc 

    def train_feat_discrepancy(self, X_s, X_t, y_s, clip = 1,mu =1):
        """
        Train phi to minimize the discrepancy

        """
        #Training
        self.train()
        self.h_disc.eval()
        parameters = nn.ModuleList([self.feature_extractor, self.model_attention, self.model_attention_target, self.h_v]).parameters() 
        
        #Feature training
        y_spred, y_sdisc, y_tpred, y_tdisc = self.forward(X_s, X_t)
        
        disc = torch.abs(self.weighted_loss(y_spred, y_sdisc)-self.target_loss_(y_tpred, y_tdisc))
        source_loss = self.weighted_loss(y_s, y_spred)
        loss = disc + mu*source_loss
        self.optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        self.optimizer.zero_grad()

        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(parameters,clip)


        self.optimizer.step()

        
        
    def train_alpha_discrepancy(self, X_s, X_t, y_s, lam_alpha=0.01, clip=1):
        """
        Train phi to minimize the discrepancy

        """
        #Training
        self.train()
        self.h_disc.eval()
        parameters = [self.alpha]
        
        #Feature training
        y_spred, y_sdisc, y_tpred, y_tdisc = self.forward(X_s, X_t)
        disc = torch.abs(self.weighted_loss(y_spred, y_sdisc)-self.target_loss_(y_tpred, y_tdisc))
        loss = disc + lam_alpha*torch.norm(self.alpha, p=2)  
        self.optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        self.optimizer.zero_grad()

        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(parameters,clip)

        self.optimizer.step()
        with torch.no_grad():
            self.alpha.clamp_(1/(self.n_views*10),1-1/(self.n_views*10))
            self.alpha.div_(torch.norm(F.relu(self.alpha), p=1))
            
    
    
    
    def weighted_loss(self, y_pred,y_s):
        loss = torch.sum(torch.stack([self.alpha[i]*self.loss(y_pred[i], y_s[i]) for i in range(self.n_views)]))
        return loss
    
    def target_loss_(self, y_pred,y_s):
        loss = torch.sum(torch.stack([self.loss(y_pred[i], y_s) for i in range(self.n_views)]))/self.n_views
        return loss
    
    def loss_views(self, y_pred,y_s):
        loss = torch.sum(torch.stack([self.loss(y_pred[i], y_s) for i in range(self.n_views)]))/self.n_views
        return loss
    
    def loss_agregation(self,y_pred):
        loss = torch.sum(torch.stack([self.alpha[i]*self.loss(y_pred[i], y_pred[j]) for i in range(self.n_views) for j in range(i,self.n_views)]))
        return loss
    
 
    def clamp(self, x):
        return torch.clamp_(x, self.min_pred, self.max_pred)
          
    
    def fit(self,X_s, X_t, y_s, y_t, X_s_val, y_val, stopping_crit, num_epochs, batch_size = 32):

        print("X_t",X_t.shape)
        
        train_dataset = E2EDatasetLoader(X_s, y_s)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        stopping_iteration = 0
        current_loss = np.inf
        # logging.info("Starting pretraining for {} epochs".format(num_epochs))
        
        # for epoch in range(num_epochs):
        #     if stopping_iteration > stopping_crit:
        #         logging.info("Stopping reached!")
        #         break
        #     losses_per_batch = []
        #     disc_per_batch = []
        #     losses_per_batch_target = []
        #     for i, (features, labels) in enumerate(dataloader):
        #         ridx = np.random.choice(X_t.shape[0], batch_size)
        #         x_bt = X_t[ridx,:]
        #         self.train_h(features,x_bt,labels)
                
        logging.info("Starting adaptation domain for {} epochs".format(num_epochs))

        for epoch in range(num_epochs):
            if stopping_iteration > stopping_crit:
                logging.info("Stopping reached!")
                break
            losses_per_batch = []
            disc_per_batch = []
            losses_per_batch_target = []
            for i, (features, labels) in enumerate(dataloader): 
                labels = [labels,labels]
                ridx = np.random.choice(X_t.shape[0], batch_size)
                x_bt = X_t[ridx,:]
                self.train_h(features,x_bt,labels)
                disc = self.train_h_discrepancy(features, x_bt, labels)
                self.train_feat_discrepancy(features,x_bt,labels)
                self.train_alpha_discrepancy(features,x_bt,labels)
                self.eval()
                y_spred, y_sdisc, y_tpred, y_tdisc = self.forward(X_s_val, X_t)
                loss = self.weighted_loss(y_spred, [y_val,y_val])
                loss_target = self.target_loss_(y_tpred, y_t)
                losses_per_batch.append(loss)
                disc_per_batch.append(disc)
                losses_per_batch_target.append(loss_target)


                
            mean_loss = torch.mean(torch.stack(losses_per_batch))
            mean_loss_disc = torch.mean(torch.stack(disc_per_batch))
            mean_loss_target = torch.mean(torch.stack(losses_per_batch_target))
            # print(self.alpha)
            if mean_loss_target < current_loss:
                current_loss = mean_loss_target
                stopping_iteration = 0
            else:
                stopping_iteration += 1
            logging.info("epoch {}, mean loss per batch source {}".format(epoch, mean_loss))
            logging.info("epoch {}, mean loss disc per batch source {}".format(epoch, mean_loss_disc))
            logging.info("epoch {}, mean loss target per batch  {}".format(epoch, mean_loss_target))
            # logging.info("epoch {}, alpha per batch source {}".format(epoch,self.alpha))

        return mean_loss, mean_loss_target
