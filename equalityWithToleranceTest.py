# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:34:51 2023

@author: chris
"""

import torch

def tensors_are_equal(tensor1, tensor2, threshold=0.9):
    # Check if the shapes of the tensors match
    if tensor1.shape != tensor2.shape:
        return False
    
    # Calculate the total number of elements in the tensors
    total_elements = tensor1.numel()

    # Calculate the number of equal elements between the tensors
    equal_elements = torch.sum(tensor1 == tensor2).item()

    # Calculate the percentage of equal elements
    percentage_equal = equal_elements / total_elements

    # Check if the percentage of equal elements meets the threshold
    return percentage_equal >= threshold

# Example tensors
tensor_a = torch.tensor([1, 2, 3,4,5,6,7,8,9,10])
tensor_b = torch.tensor([1, 2, 3,4,5,6,7,8,9,11])

# Check if tensors are equal with at least 90% matching values
if tensors_are_equal(tensor_a, tensor_b, threshold=0.9):
    print("equal")
else:
    print("not ")
