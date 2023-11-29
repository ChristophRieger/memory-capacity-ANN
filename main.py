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
import matplotlib.pyplot as plt

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
      self.fullyConnectedLayer = torch.nn.Linear(inputSize, outputSize)
      self.activation = torch.nn.Sigmoid()
      # !!! This has to be called last, otherwise it doesn't move the parameters
      # above to the device
      self.to(my_device) # move the model to the same device as tensors

  def forward(self, x):
      x = self.fullyConnectedLayer(x)
      x = self.activation(x)
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
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
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
plt.close("all")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# parameters
N = 400
sparsity = 0.1 # fraction of active bits in data
# dataset_size = 2
my_batch_size = 1
# EPOCHS = 2
learning_rate = 0.01
# momentum = 0.9

dataset_sizes = [200, 220, 240, 260, 280, 300]

for dataset_size in dataset_sizes:

  # Command Center
  load_model = False
  model_state_path = 'modelStates/N{}_s{}_dS{}_lr{}_bS{}_{}'.format(N, sparsity, dataset_size, learning_rate, my_batch_size, timestamp)
  results_path =         'results/N{}_s{}_dS{}_lr{}_bS{}_{}'.format(N, sparsity, dataset_size, learning_rate, my_batch_size, timestamp)
  
  if torch.cuda.is_available():
      my_device = torch.device('cuda')
  else:
      my_device = torch.device('cpu')
  print('Device: {}'.format(my_device))
  
  X, y = create_dataset(N, sparsity, dataset_size) 
  
  dataset_train = customTensorDataset(X, y, my_device)
  train_loader = DataLoader(dataset=dataset_train, batch_size=my_batch_size, shuffle=True)
  oneLayerModel = OneLayerModel(N, N, my_device)
  if load_model:
    oneLayerModel.load_state_dict(torch.load('path_to_model'))
  
  # Optimizers specified in the torch.optim package
  # optimizer = torch.optim.SGD(oneLayerModel.parameters(), lr=learning_rate, momentum=momentum)
  optimizer = torch.optim.Adam(oneLayerModel.parameters(), lr=learning_rate)
  # I try Binary Cross Entropy Loss...
  loss_fn = torch.nn.BCELoss()
  # !!! TODO There is BCEWITHLOGITSLOSS which combines this loss with also applying 
  # the sigmoid function, and it says its prefered to use the 2 separately...
  
  # Initializing in a separate cell so we can easily add more epochs to the same run
  writer = SummaryWriter(results_path)
  epoch_number = 0
  
  loss_per_epoch = []
  best_loss = 1_000_000.
  avg_loss = 0
  if not os.path.exists('modelStates'):
    os.mkdir('modelStates')
  if not os.path.exists(model_state_path):
    os.mkdir(model_state_path)
  # for epoch in range(EPOCHS):
  while True:
      print('EPOCH {}:'.format(epoch_number + 1))
  
      # Make sure gradient tracking is on, and do a pass over the data
      oneLayerModel.train(True)
      avg_loss = train_one_epoch(epoch_number, train_loader, oneLayerModel, optimizer, loss_fn, my_batch_size, my_device, writer)
  
      # Set the model to evaluation mode, disabling dropout and using population
      # statistics for batch normalization.
      oneLayerModel.eval()
              
      print('LOSS train {}'.format(avg_loss))
      loss_per_epoch.append(avg_loss)
      writer.add_scalars('Training loss',
                      { 'Training loss' : avg_loss },
                      epoch_number + 1)
      writer.flush()
  
      if best_loss - avg_loss < 10 ** -6 and epoch_number > 100:
        torch.save(oneLayerModel.state_dict(), model_state_path + '/model_{}'.format(epoch_number))
        break
      if epoch_number > 500:
        torch.save(oneLayerModel.state_dict(), model_state_path + '/model_{}'.format(epoch_number))
        break
      # Track best performance, and save the model's state
      if avg_loss < best_loss:
          best_loss = avg_loss
  
      epoch_number += 1
  if not os.path.exists('results'):
    os.mkdir('results')
  if not os.path.exists(results_path):
    os.mkdir(results_path)
  np.savetxt(results_path + "/X.txt", X, fmt='%d')
  np.savetxt(results_path + "/y.txt", y, fmt='%d')
  
  # calculate percentage of correctly learned patterns
  correct_y_predictions = np.zeros(((1,1)))
  for i, data in enumerate(train_loader):
    inputs, labels = data
    y_prediction = oneLayerModel(inputs)
    for j in range(y_prediction.size()[1]):
      if y_prediction[0][j] >= 0.5:
        y_prediction[0][j] = 1
      else:
        y_prediction[0][j] = 0
    if torch.equal(y_prediction, labels):
      correct_y_predictions[0] += 1
      
  percentage_of_correct_memorizations = correct_y_predictions[0] / len(y)
  np.savetxt(results_path + "/percentage_of_patterns_memorized.txt", percentage_of_correct_memorizations, fmt='%d')
  
  
  plt.figure()  
  plt.plot(loss_per_epoch)
  plt.title("Training loss ({}% accuracy)".format(percentage_of_correct_memorizations[0]*100), fontsize=14)
  plt.ylabel("Binary Cross Entropy", fontsize=12)
  plt.xlabel("Epoch", fontsize=12)
  plt.tick_params(axis='both', labelsize=11)
  plt.savefig(results_path + "/trainingPlot" + ".svg")  
  plt.savefig(results_path + "/trainingPlot" + ".png")
    