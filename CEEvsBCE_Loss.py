# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:43:12 2023

@author: chris
"""
import torch
import numpy as np

label = torch.tensor(np.array((1, 0, 0), dtype='float32'))
prediction = torch.tensor(np.array((1, 0, 0), dtype='float32'))

CEE_loss_fn = torch.nn.CrossEntropyLoss()
BCE_loss_fn = torch.nn.BCELoss()

CEE_loss = CEE_loss_fn(prediction, label)
BCE_loss = BCE_loss_fn(prediction, label)

print("CEE loss: " + str(CEE_loss.item()))
print("BCE loss: " + str(BCE_loss.item()))
