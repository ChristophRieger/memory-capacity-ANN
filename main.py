# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 11:16:22 2023

@author: chris
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random

# class to build a dataset
class customTensorDataset(Dataset):
  def __init__(self, X, y, my_device):
    tensors = (torch.tensor(X, device=my_device), torch.tensor(y, device=my_device))
    assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
    self.tensors = tensors

  def __getitem__(self, index):
    return self.tensors[0][index], self.tensors[1][index]

  def __len__(self):
    return self.tensors[0].size(0)

class OneLayerModel(torch.nn.Module):

  def __init__(self, inputSize, outputSize, my_device):
      super(OneLayerModel, self).__init__()
      # self.inputLayer = torch.nn.Linear(inputSize, inputSize)
      self.fullyConnectedLayer = torch.nn.Linear(inputSize, outputSize)
      self.activation = torch.nn.Sigmoid()
      self.softmax = torch.nn.Softmax()
      # !!! This has to be called last, otherwise it doesn't move the parameters
      # above to the device
      self.to(my_device) # move the model to the same device as tensors

  def forward(self, x):
      # x = self.inputLayer(x)
      # x = self.activation(x)
      x = self.fullyConnectedLayer(x)
      x = self.activation(x)
      x = self.softmax(x)
      return x

def train_one_epoch(epoch_index, training_loader, model, optimizer, loss_fn, my_device, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        # fix (this was only needed locally, on cluster it worked) acc. to https://stackoverflow.com/questions/69742930/runtimeerror-nll-loss-forward-reduce-cuda-kernel-2d-index-not-implemented-for
        labels = labels.type(torch.LongTensor)
        labels = labels.to(my_device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        # print(model)
        outputs = model(inputs)

        # Compute the loss and its gradients
        # output has size of (nOutput, batchSize)
        # print("output: ")
        # print(outputs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def create_dataset(N, sparsity, size):
  X = np.zeros((size, N))
  y = np.zeros((size, N))
  bits = list(range(0, N))
  active_bits = int(N * sparsity)
  for i in range(size):
    X_choices = random.sample(bits, active_bits)
    X[i, X_choices] = 1 
    y_choices = random.sample(bits, active_bits)
    y[i, y_choices] = 1
    
  return X, y

# START

if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))

N = 100
sparsity = 0.1 # fraction of active bits in data
dataset_size = 10000

X, y = create_dataset(N, sparsity, dataset_size) 

# I set train and validation dataset equal, as I need the same random data to
# check if a pattern was memorized
dataset_train = customTensorDataset(X, y, my_device)
dataset_validation = customTensorDataset(X, y, my_device)
train_loader = DataLoader(dataset=dataset_train, batch_size=3, shuffle=True)
validation_loader = DataLoader(dataset=dataset_validation, batch_size=3, shuffle=False)
# for index, data in enumerate(train_loader):
  # access input vector (dont know why we need double 0... its in double square brackets..)
  # print(data[0][0])  
  # access value of input vector
  # print(data[0][0][2])
  # access label
  # print(data[1][0])
    
oneLayerModel = OneLayerModel(10, 10, my_device)
# print('The model:')
# print(oneLayerModel)
# print('The parameters:')
# for param in oneLayerModel.parameters():
  # print(param)
  
# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(oneLayerModel.parameters(), lr=0.001, momentum=0.9)

loss_fn = torch.nn.CrossEntropyLoss()

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    oneLayerModel.train(True)
    avg_loss = train_one_epoch(epoch_number, train_loader, oneLayerModel, optimizer, loss_fn, my_device, writer)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    oneLayerModel.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            # fix (this was only needed locally, on cluster it worked) acc. to https://stackoverflow.com/questions/69742930/runtimeerror-nll-loss-forward-reduce-cuda-kernel-2d-index-not-implemented-for
            vlabels = vlabels.type(torch.LongTensor)
            vlabels = vlabels.to(my_device)
            
            voutputs = oneLayerModel(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
            
    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(oneLayerModel.state_dict(), model_path)

    epoch_number += 1