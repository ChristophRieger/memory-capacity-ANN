# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 11:16:22 2023

@author: chris
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

class customTensorDataset(Dataset):
  def __init__(self, X, y):
    # [X, y] = dataset
    tensors = (torch.tensor(X), torch.tensor(y))
    assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
    self.tensors = tensors

  def __getitem__(self, index):
    return self.tensors[0][index], self.tensors[1][index]

  def __len__(self):
    return self.tensors[0].size(0)





features = [[0,0,1],[0,1,0], [1,0,0]]
labels = [1, 2, 3]

# wir haben ein map-style dataset... 
dataset_train = customTensorDataset(features, labels)
trainloader = DataLoader(dataset=dataset_train, shuffle=True)
for index, data in enumerate(trainloader):
    print(data)
# test = DataLoader(features, labels)
# test = TensorDataset(features, labels)