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
import sys
import math
import os

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
      # self.softmax = torch.nn.Softmax()
      # !!! This has to be called last, otherwise it doesn't move the parameters
      # above to the device
      self.to(my_device) # move the model to the same device as tensors

  def forward(self, x):
      # x = self.inputLayer(x)
      # x = self.activation(x)
      x = self.fullyConnectedLayer(x)
      x = self.activation(x)
      # x = self.softmax(x)
      return x

def train_one_epoch(epoch_index, training_loader, model, optimizer, loss_fn, my_batch_size, my_device, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        # fix (this was only needed locally, on cluster it worked) acc. to https://stackoverflow.com/questions/69742930/runtimeerror-nll-loss-forward-reduce-cuda-kernel-2d-index-not-implemented-for
        # now not needed anymore XD
        # labels = labels.type(torch.LongTensor)
        # labels = labels.to(my_device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        # print(model)
        outputs = model(inputs)

        # Compute the loss and its gradients
        # output has size of (nOutput, batchSize)
        # print("inputs")
        # print(inputs)
        # print("outputs")
        # print(outputs)
        # print("labels")
        # print(labels)
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        # I commented code below out, as I do not need the loss within one epoch..
        # if i % my_batch_size == my_batch_size - 1:
        #     last_loss = running_loss / my_batch_size # loss per batch
        #     # print('  batch {} loss: {}'.format(i + 1, last_loss))
        #     tb_x = epoch_index * len(training_loader) + i + 1
        #     tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        #     running_loss = 0.
    last_loss = running_loss
    return last_loss / len(training_loader)

def create_dataset(N, sparsity, size):
  print("Generating dataset...")
  X = np.zeros((size, N), dtype='float32')
  y = np.zeros((size, N), dtype='float32')
  bits = list(range(0, N))
  active_bits = int(N * sparsity)
  inactive_bits = int(N * (1 - sparsity))
  
  # check if size is bigger than the amount of possibilities
  possible_permutations = math.factorial(N) / (math.factorial(active_bits) * math.factorial(inactive_bits))
  if (size > possible_permutations):
    sys.exit("The given size of " + str(size) + " is bigger than the amount of possible permutations of " + str(int(possible_permutations)))

  for i in range(size):
    X_choices = random.sample(bits, active_bits)
    X_vector_to_check = np.zeros((1, N))
    X_vector_to_check[0, X_choices] = 1 
    while(any((X==X_vector_to_check).all(1))):
      X_choices = random.sample(bits, active_bits)
      X_vector_to_check = np.zeros((1, N))
      X_vector_to_check[0, X_choices] = 1 
    X[i, :] = X_vector_to_check
      
    y_choices = random.sample(bits, active_bits)
    y_vector_to_check = np.zeros((1, N))
    y_vector_to_check[0, y_choices] = 1 
    while(any((y==y_vector_to_check).all(1))):
      y_choices = random.sample(bits, active_bits)
      y_vector_to_check = np.zeros((1, N))
      y_vector_to_check[0, y_choices] = 1 
    y[i, :] = y_vector_to_check
    
  print("Dataset generated!")
  return X, y

# START

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# parameters
N = 20
sparsity = 0.1 # fraction of active bits in data
dataset_size = 40
my_batch_size = 1
EPOCHS = 3000
learning_rate = 0.01
momentum = 0.9

# Command Center
load_model = False
model_state_path = 'modelStates/N{}_s{}_dS{}_lr{}_m{}_bS{}_E{}_{}'.format(N, sparsity, dataset_size, learning_rate, momentum, my_batch_size, EPOCHS, timestamp)
results_path =         'results/N{}_s{}_dS{}_lr{}_m{}_bS{}_E{}_{}'.format(N, sparsity, dataset_size, learning_rate, momentum, my_batch_size, EPOCHS, timestamp)

if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))



X, y = create_dataset(N, sparsity, dataset_size) 

# sys.exit()
# print("X")
# print(X)
# print("y")
# print(y)
# I set train and validation dataset equal, as I need the same random data to
# check if a pattern was memorized
dataset_train = customTensorDataset(X, y, my_device)
dataset_validation = customTensorDataset(X, y, my_device)
train_loader = DataLoader(dataset=dataset_train, batch_size=my_batch_size, shuffle=True)
validation_loader = DataLoader(dataset=dataset_validation, batch_size=1, shuffle=False)
# for index, data in enumerate(train_loader):
  # access input vector (dont know why we need double 0... its in double square brackets..)
  # print(data[0][0])  
  # access value of input vector
  # print(data[0][0][2])
  # access label
  # print(data[1][0])
oneLayerModel = OneLayerModel(N, N, my_device)
if load_model:
  oneLayerModel.load_state_dict(torch.load('path_to_model'))
# print('The model:')
# print(oneLayerModel)
# print('The parameters initially:')
# for param in oneLayerModel.parameters():
#   print(param)

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(oneLayerModel.parameters(), lr=learning_rate, momentum=momentum)

# it seems this is the wrong loss function, as it is not supposed to be used
# for multiple binary classes
# loss_fn = torch.nn.CrossEntropyLoss()

# I try Binary Cross Entropy Loss...
loss_fn = torch.nn.BCELoss()
# !!! TODO There is BCEWITHLOGITSLOSS which combines this loss with also applying 
# the sigmoid function, and it says its prefered to use the 2 separately...

# Loss Debugging
# outputs = np.zeros((1, 10), dtype='float32')
# outputs[0,0] = 1
# print("my outputs")
# print(outputs)
# labels = np.zeros((1, 10), dtype='float32')
# labels[0,0] = 0.8
# print("my labels")
# print(labels)
# outputss = torch.tensor(outputs, device=my_device)
# labelss = torch.tensor(labels, device=my_device)

# print("myloss")
# print(loss_fn(outputss, labelss))

# Initializing in a separate cell so we can easily add more epochs to the same run
writer = SummaryWriter(results_path)
epoch_number = 0

best_vloss = 1_000_000.
if not os.path.exists('modelStates'):
  os.mkdir('modelStates')
if not os.path.exists(model_state_path):
  os.mkdir(model_state_path)
for epoch in range(EPOCHS):
    # print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    oneLayerModel.train(True)
    avg_loss = train_one_epoch(epoch_number, train_loader, oneLayerModel, optimizer, loss_fn, my_batch_size, my_device, writer)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    oneLayerModel.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            # fix (this was only needed locally, on cluster it worked) acc. to https://stackoverflow.com/questions/69742930/runtimeerror-nll-loss-forward-reduce-cuda-kernel-2d-index-not-implemented-for
            # now not needed anymore XD
            # vlabels = vlabels.type(torch.LongTensor)
            # vlabels = vlabels.to(my_device)
            
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
        torch.save(oneLayerModel.state_dict(), model_state_path + '/model_{}'.format(epoch_number))

    epoch_number += 1
if not os.path.exists('results'):
  os.mkdir('results')
if not os.path.exists(results_path):
  os.mkdir(results_path)
np.savetxt(results_path + "/X.txt", X, fmt='%d')
np.savetxt(results_path + "/y.txt", y, fmt='%d')

# calculate percentage of correctly learned patterns
correct_y_predictions = np.zeros(((1,1)))
for i, vdata in enumerate(validation_loader):
  vinputs, vlabels = vdata
  y_prediction = oneLayerModel(vinputs)
  for j in range(y_prediction.size()[1]):
    if y_prediction[0][j] >= 0.5:
      y_prediction[0][j] = 1
    else:
      y_prediction[0][j] = 0
  # this returns a vector, but I want a single bool
  # if y_prediction[0] == vlabels[0]:
  if torch.equal(y_prediction, vlabels):
    correct_y_predictions[0] += 1
    
percentage_of_correct_memorizations = correct_y_predictions[0] / len(y)
np.savetxt(results_path + "/percentage_of_patterns_memorized.txt", percentage_of_correct_memorizations, fmt='%d')

  