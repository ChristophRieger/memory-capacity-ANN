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

      self.linear1 = torch.nn.Linear(inputSize, outputSize)
      self.activation = torch.nn.Sigmoid()
      self.to(my_device) # move the model to the same device as tensors
      # self.softmax = torch.nn.Softmax()
      # nn.CrossEntropyLoss

  def forward(self, x):
      x = self.linear1(x)
      x = self.activation(x)
      # x = self.softmax(x)
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
        # print(inputs)
        # fix acc. to https://stackoverflow.com/questions/69742930/runtimeerror-nll-loss-forward-reduce-cuda-kernel-2d-index-not-implemented-for
        labels = labels.type(torch.LongTensor)
        labels = labels.to(my_device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        # print(model)
        outputs = model(inputs)

        # Compute the loss and its gradients
        # print(outputs)
        loss = loss_fn(outputs, labels)
        print(loss)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))

X = np.array([[1,1,1,0,0,0,0,0,0],
     [0,0,0,1,1,1,0,0,0],
     [0,0,0,0,0,0,1,1,1]], dtype='float32')
# X = [[1,1,1,0,0,0,0,0,0],
#      [0,0,0,1,1,1,0,0,0],
#      [0,0,0,0,0,0,1,1,1]]
# image = np.array([[255, 255, 0, 0, 0, 255, 255, 255, 255]], dtype=np.uint8)

y = np.array([1, 2, 3])

# wir haben ein map-style dataset... 
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
    
oneLayerModel = OneLayerModel(9, 3, my_device)
# print('The model:')
# print(oneLayerModel)
# print('The parameters:')
# for param in oneLayerModel.parameters():
  # print(param)
  
  
# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(oneLayerModel.parameters(), lr=0.001, momentum=0.9)
  
  
loss_fn = torch.nn.CrossEntropyLoss()
# NB: Loss functions expect data in batches, so we're creating batches of 4
# Represents the model's confidence in each of the 10 classes for a given input
dummy_outputs = torch.rand(4, 10)
# Represents the correct class among the 10 being tested
dummy_labels = torch.tensor([1, 5, 3, 7])
print(dummy_outputs)
print(dummy_labels)
loss = loss_fn(dummy_outputs, dummy_labels)
print('Total loss for this batch: {}'.format(loss.item()))


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
    print('after train one epoch')

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    oneLayerModel.eval()
    print('after eval')

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
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