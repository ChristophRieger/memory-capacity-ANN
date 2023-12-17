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
      # This has to be called last, otherwise it doesn't move the parameters
      # above to the device
      self.to(my_device) # move the model to the same device as tensors

  def forward(self, x):
      x = self.fullyConnectedLayer(x)
      x = self.activation(x)
      return x

def train_one_epoch(epoch_index, training_loader, model, optimizer, loss_fn, my_batch_size, my_device, tb_writer, use_custom_loss):
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
        if use_custom_loss:
          loss = loss_fn(outputs, labels)
          loss_indices = torch.arange(0, N, device=my_device)
        
          # set the bits to ignore
          #### get all indices where labels == 1
          one_labels = torch.nonzero(labels == 1)[:, 1]
          # get the losses where labels == 1
          loss_ones = loss[0, one_labels]
          # get the indices where the losses where in the unfiltered loss
          original_loss_indices_ones = loss_indices[one_labels]
          
          # !!!!! sorting is veeery slow
          # sort the loss of all labels == 1
          # sorted_loss_ones, new_indices_of_loss_ones = torch.sort(loss_ones, descending=True)
          # set #bits_to_ignore of the highest losses to 0
          # for ignore_iterator in range(bits_to_ignore):
          #   loss[0, original_loss_indices_ones[new_indices_of_loss_ones[ignore_iterator]]] = 0
          
          # !!! bc of that trying to use topk instead: (should be faster)
          loss_ones_to_remove, indices_of_loss_ones_to_remove = torch.topk(loss_ones, bits_to_ignore, sorted = False)
          # set #bits_to_ignore of the highest losses to 0
          for ignore_iterator in range(len(indices_of_loss_ones_to_remove)):
            loss[0, original_loss_indices_ones[indices_of_loss_ones_to_remove[ignore_iterator]]] = 0
          
          ### get all indices where labels == 0
          zero_labels = torch.nonzero(labels == 0)[:, 1]
          # get the losses where labels == 1
          loss_zeros = loss[0, zero_labels]
          # get the indices where the losses where in the unfiltered loss
          original_loss_indices_zeros = loss_indices[zero_labels]
          # sort the loss of all labels == 0
          # sorted_loss_zeros, new_indices_of_loss_zeros = torch.sort(loss_zeros, descending=True)
          # set #bits_to_ignore of the highest losses to 0
          # for ignore_iterator in range(bits_to_ignore):
            # loss[0, original_loss_indices_zeros[new_indices_of_loss_zeros[ignore_iterator]]] = 0
          loss_zeros_to_remove, indices_of_loss_zeros_to_remove = torch.topk(loss_zeros, bits_to_ignore, sorted = False)
          # set #bits_to_ignore of the highest losses to 0
          for ignore_iterator in range(len(indices_of_loss_zeros_to_remove)):
            loss[0, original_loss_indices_zeros[indices_of_loss_zeros_to_remove[ignore_iterator]]] = 0
          
          ### unify the loss again into a single number
          loss = torch.mean(loss)
        else:
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
  if (N < 101):
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

# Command center
use_custom_loss = True

# parameters
N = 20
max_epochs = 300
variance_runs = 10
sparsity = 0.1 # fraction of active bits in data
tolerance = 0.1 # how many bits of the active bits are ignored during training and validation. Also how many inactive bits are ignored
bits_to_ignore = int(N * sparsity * tolerance)

# dataset_size = 2
my_batch_size = 1
# EPOCHS = 2
learning_rate = 0.01
# momentum = 0.9

dataset_sizes = [30, 35, 40, 45]

for dataset_size in dataset_sizes:
  accuracies = []
  for variance_iterator in range(variance_runs):
    # Command Center
    load_model = False
    model_state_path = 'modelStates/N{}_s{}_dS{}_lr{}_bS{}_{}'.format(N, sparsity, dataset_size, learning_rate, my_batch_size, timestamp)
    results_path =         'results/N{}_s{}_dS{}_lr{}_bS{}_{}'.format(N, sparsity, dataset_size, learning_rate, my_batch_size, timestamp)
    
    if torch.cuda.is_available():
        my_device = torch.device('cuda')
    else:
        my_device = torch.device('cpu')
    # my_device = torch.device('cpu')
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
  
    if use_custom_loss:
      loss_fn = torch.nn.BCELoss(reduction='none')
    else:
      loss_fn = torch.nn.BCELoss()
    # !!! TODO There is BCEWITHLOGITSLOSS which combines this loss with also applying 
    # the sigmoid function, and it says its prefered to use the 2 separately...
    
    # Initializing in a separate cell so we can easily add more epochs to the same run
    writer = SummaryWriter(results_path)
    epoch_number = 0
    
    loss_per_epoch = []
    best_loss = 1_000_000.
    avg_loss = 0
    percentage_of_correct_memorizations_list = []
    if not os.path.exists('modelStates'):
      os.mkdir('modelStates')
    if not os.path.exists(model_state_path):
      os.mkdir(model_state_path)
    # for epoch in range(EPOCHS):
    while True:
        print('EPOCH {}:'.format(epoch_number + 1))
    
        # Make sure gradient tracking is on, and do a pass over the data
        oneLayerModel.train(True)
        avg_loss = train_one_epoch(epoch_number, train_loader, oneLayerModel, optimizer, loss_fn, my_batch_size, my_device, writer, use_custom_loss)
    
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        oneLayerModel.eval()
                
        print('LOSS train {}'.format(avg_loss))
        loss_per_epoch.append(avg_loss)
        
        # calculate percentage of correctly learned patterns
        # correct_y_predictions = np.zeros(((1,1)))
        # for i, data in enumerate(train_loader):
        #   print(str(i))
        #   inputs, labels = data
        #   y_prediction = oneLayerModel(inputs)
        #   for j in range(y_prediction.size()[1]):
        #     if y_prediction[0][j] >= 0.5:
        #       y_prediction[0][j] = 1
        #     else:
        #       y_prediction[0][j] = 0
        #   if torch.equal(y_prediction, labels):
        #     correct_y_predictions[0] += 1
        # percentage_of_correct_memorizations_list.append(correct_y_predictions[0] / len(y))
        # writer.add_scalars('Training loss',
        #                 { 'Training loss' : avg_loss },
        #                 epoch_number + 1)
        # writer.flush()
    
        # if best_loss - avg_loss < 10 ** -6 and epoch_number > 50:
          # torch.save(oneLayerModel.state_dict(), model_state_path + '/model_{}'.format(epoch_number))
          # break
        if epoch_number > max_epochs:
          torch.save(oneLayerModel.state_dict(), model_state_path + '/model_{}'.format(epoch_number))
          break
        # if percentage_of_correct_memorizations_list[-1][0] == 1:
        #   torch.save(oneLayerModel.state_dict(), model_state_path + '/model_{}'.format(epoch_number))
        #   break
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
      if use_custom_loss:
        
        one_labels = torch.nonzero(labels == 1)[:, 1]
        one_label_values = labels[0, one_labels]
        y_prediction_ones = y_prediction[0, one_labels]
        zero_labels = torch.nonzero(labels == 0)[:, 1]
        zero_label_values = labels[0, zero_labels]
        y_prediction_zeros = y_prediction[0, zero_labels]
        
        equal_bits_ones = torch.sum(y_prediction_ones == one_label_values).item()
        equal_bits_zeros = torch.sum(y_prediction_zeros == zero_label_values).item()
        
        if equal_bits_ones >= N * sparsity -  bits_to_ignore and equal_bits_zeros >= N - N * sparsity - bits_to_ignore:
          correct_y_predictions[0] += 1
      else:
          if torch.equal(y_prediction, labels):
            correct_y_predictions[0] += 1
        
    percentage_of_correct_memorizations = correct_y_predictions[0] / len(y)
    accuracies.append(percentage_of_correct_memorizations)
    # np.savetxt(results_path + "/percentage_of_patterns_memorized.txt", percentage_of_correct_memorizations, fmt='%d')
    
    
    plt.figure()  
    plt.plot(loss_per_epoch)
    plt.title("Training loss ({}% accuracy)".format(percentage_of_correct_memorizations[0]*100), fontsize=14)
    plt.ylabel("Binary Cross Entropy", fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.tick_params(axis='both', labelsize=11)
    # plt.savefig(results_path + "/trainingPlot{}".format(variance_iterator) + ".svg")  
    plt.savefig(results_path + "/trainingPlot{}".format(variance_iterator) + ".png")
  
  mean_accuracy = np.mean(accuracies)
  variance = np.var(accuracies)
  mean_patterns_memorized = dataset_size * mean_accuracy
  np.savetxt(results_path + "/{}patterns_{}accu_{}var_{}varRuns.txt".format(mean_patterns_memorized, mean_accuracy, variance, variance_runs), [12345], fmt='%d')

  
    