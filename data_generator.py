import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
from feature import calculate_feature_for_all_audio_files
import numpy as np
import pandas as pd
import pickle
import os
import config



def read_pkl(path):

    f=open(path,'rb')
    data = pickle.load(f)
    return data

def data_generate():
    train_data = read_pkl('./Train_feature_dict.pkl')
    test_data = read_pkl('./Test_feature_dict.pkl')

    x_train = train_data['feature']
    y_train = train_data['label']
    x_test = test_data['feature']
    y_test = test_data['label']
    y_train = np.array(pd.get_dummies(y_train))
    y_test = np.array(pd.get_dummies(y_test))

    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    x_train, y_train, x_test, y_test = map(torch.tensor, (x_train, y_train, x_test, y_test))

    x_train = torch.unsqueeze(x_train, dim=1).float()
    x_test = torch.unsqueeze(x_test, dim=1).float()

    return x_train,y_train,x_test, y_test

x_train,y_train,x_test, y_test= data_generate()

def get_data(x_train, y_train, x_test, y_test, bs):

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(test_ds, batch_size=bs * 2),
    )