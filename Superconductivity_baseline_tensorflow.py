# -*- coding: utf-8 -*-

import os
    

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import clone_model
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import Model


import pandas as pd
import numpy as np
import random
from numpy.random import default_rng

from utils.utils import download_uci, superconduct, create_views, create_domain, zero_padding, domain
from utils.utils import kin, BaggingModels, cross_val, domain, download_uci, superconduct, create_views, zero_padding
from utils.utils_hdisc import batch_loader, split_source_target, val_split

## utils preprocessing from sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler



from sklearn.metrics import mean_squared_error


##import model
from models.WANN import WANN
from adapt.instance_based import KLIEP, KMM, TrAdaBoostR2
from adapt.feature_based import DANN,MDD, ADDA, DeepCORAL

from warnings import filterwarnings
filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



# Set seed
seed = 1
tf.random.experimental.Generator.from_seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)




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

## for each domain we create diffenrent views
rng = default_rng()
SelectFeatures = rng.choice((X_v1.shape[1] + X_v2.shape[1]), size=int((X_v1.shape[1] + X_v2.shape[1])/2), replace=False) ## we select randomly features
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
    
        
def get_base_model(shape=84, activation=None, C=1, name="BaseModel"):
    inputs = Input(shape=(shape,))
    modeled = Dense(256, activation='relu',
                          kernel_constraint=MinMaxNorm(0, C),
                          bias_constraint=MinMaxNorm(0, C))(inputs)
    modeled = Dropout(0.5)(modeled)
    modeled = Dense(100, activation='relu',
                          kernel_constraint=MinMaxNorm(0, C),
                          bias_constraint=MinMaxNorm(0, C))(modeled)
    modeled = Dropout(0.2)(modeled)
    modeled = Dense(1, activation=activation,
                    kernel_constraint=MinMaxNorm(0, C),
                    bias_constraint=MinMaxNorm(0, C))(modeled)
    model = Model(inputs, modeled, name=name)
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    return model


# def get_base_model(shape=84, activation=None, C=1., name="BaseModel"):
#     inputs = Input(shape=(shape,))
#     modeled = Dense(100, activation='relu',
#                           kernel_constraint=MinMaxNorm(0, C),
#                           bias_constraint=MinMaxNorm(0, C))(inputs)
#     modeled = Dense(100, activation='relu',
#                     kernel_constraint=MinMaxNorm(0, C),
#                     bias_constraint=MinMaxNorm(0, C))(modeled)
#     modeled = Dense(1, activation=activation,
#                     kernel_constraint=MinMaxNorm(0, C),
#                     bias_constraint=MinMaxNorm(0, C))(modeled)
#     model = Model(inputs, modeled, name=name)
#     model.compile(optimizer=Adam(0.0001), loss='mean_squared_error')
#     return model


def get_encoder(shape=84, C=1, name="encoder"):
    inputs = Input(shape=(shape,))
    modeled = Dense(100, activation='relu',
                          kernel_constraint=MinMaxNorm(0, C),
                          bias_constraint=MinMaxNorm(0, C))(inputs)
    model = Model(inputs, modeled)
    model.compile(optimizer="adam", loss='mean_squared_error')
    return model


def get_task(shape=100, C=1, activation=None, name="task"):
    inputs = Input(shape=(shape,))
    modeled = Dense(100, activation='relu',
                    kernel_constraint=MinMaxNorm(0, C),
                    bias_constraint=MinMaxNorm(0, C))(inputs)
    modeled = Dense(1, activation=activation,
                          kernel_constraint=MinMaxNorm(0, C),
                          bias_constraint=MinMaxNorm(0, C))(modeled)
    model = Model(inputs, modeled)
    model.compile(optimizer="adam", loss='mean_squared_error')
    return model



for percent in [0.7]:
    Score_target = {}
    Score_source = {}
    for source in [0, 1, 2, 3]:
        for target in [0, 1, 2, 3]:
            if source != target:
                
    
                print("------------------------------------------------------------Source Doamin:",source)
                print("------------------------------------------------------------Target Doamin:",target)
                
                
                for method in ["WANN", "KLIEP", "KMM", "DANN", "ADDA", "DeepCORAL", "MDD", "TrAdaBoostR2"]:
                    print(method)
                    
                    if not method in Score_target:
                        Score_target[method] = []
                        Score_source[method] = []
                    
            

                        
                    X_t = np.concatenate([X_s[target][0][dataframe_views1].to_numpy(),X_s[target][1][dataframe_views2].to_numpy()],1)
                    X_sv, X_unlabeled, y_s, y_unlabeled = val_split([X_s[source][0].drop(dataframe_views1, axis=1).to_numpy(),X_s[source][1].drop(dataframe_views2, axis=1).to_numpy()], [y_D[source],y_D[source]], frac_train = percent)
                    X_Dv = np.concatenate((X_sv),1)
                    X_unlabeled = np.concatenate((X_unlabeled),1)
                    # X_c = np.delete(X_Dv,SelectFeatures, 1)
                    X = np.concatenate((X_Dv,X_t))
                    y = np.concatenate((y_s[0],y_D[target]))
                    src_index = np.array(range(len(y_s[0])))
                    tgt_index = np.array(range(len(y_s[0]), len(y_s[0]) + len(y_D[target])))
                    tgt_index_labeled = np.random.choice(tgt_index,10, replace=False)
                    train_index = np.concatenate((src_index, tgt_index_labeled))
                            
                        
                    if method == "WANN":
                        model = WANN(get_base_model=get_base_model, C=1, optimizer=Adam(0.01)) 
                        model.fit(X, y, [train_index, tgt_index_labeled], epochs=350, verbose=0, batch_size=128)
            
                                    
                    elif method == "TrAdaBoostR2":
                        model = TrAdaBoostR2(get_base_model(), verbose=0, random_state=seed)
                        model.fit(X[train_index], y[train_index], X[tgt_index_labeled], y[tgt_index_labeled],
                                      epochs=250, batch_size=128, verbose=0)
                            
                    elif method in ["DANN","KLIEP", "KMM", "ADDA", "DeepCORAL", "MDD"]:
        
                            
                        if method == "DANN":
                            model = DANN(encoder=get_encoder(), task=get_task(), random_state=seed,
                                         discriminator=get_task(activation="sigmoid"),
                                         optimizer=Adam(0.001), lambda_=0.1, loss="mse")
    
                                
                        if method == "DeepCORAL":
                            model = DeepCORAL(encoder=get_encoder(), task=get_task(), lambda_=10.,
                                              optimizer=Adam(0.001), loss="mse",
                                              random_state=seed)
                            model.fit(X[train_index], y[train_index], X[tgt_index], epochs=250, batch_size=128, verbose=0)
                        if method == "MDD":
                            model = MDD(encoder=get_encoder(), task=get_task(), random_state=seed,
                                        optimizer=Adam(0.001), lambda_=0.001, loss="mse")
    
                            
                        if method == "ADDA":
                            encoder = get_encoder()
                            task=get_task()
                            discriminator=get_task(activation="sigmoid")
                            dann = DANN(encoder, task, discriminator, loss="mse", copy=False,
                                        lambda_=0., random_state=seed)
                            dann.fit(X[train_index], y[train_index], X[tgt_index], epochs=250, batch_size=128, verbose=0)
                            model = ADDA(encoder=encoder, task=task,
                                         discriminator=discriminator, random_state=seed,
                                         optimizer=Adam(0.001), loss="mse")
    
                                
                        if method == "KLIEP":
                            model = KLIEP(get_base_model(), sigmas=0.001, random_state=seed)
    
                                
                        if method == "KMM":
                            model = KMM(get_base_model(),verbose=0, random_state=seed)
                            
                            
                            
                        model.fit(X[train_index], y[train_index], X[tgt_index], epochs=250, batch_size=128, verbose=0)
            
                
                    err_s = np.mean(np.square(np.subtract(model.predict(X_unlabeled).ravel(), y_unlabeled[0])))
                    err_t =  np.mean(np.square(np.subtract(model.predict(X).ravel()[tgt_index], y[tgt_index])))
                    print(err_t)
                    print(err_s)
                    Score_target[method].append(err_t)
                    Score_source[method].append(err_s)
                        
                        
                pd.DataFrame(Score_target).to_csv("./dataset/results/Superconductivity_target"+".csv")
                pd.DataFrame(Score_source).to_csv("./dataset/results/Superconductivity_source"+".csv")
