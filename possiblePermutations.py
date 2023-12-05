# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:38:01 2023

@author: chris
"""
import math
import matplotlib.pyplot as plt
import numpy as np

N =                  [10, 20, 30, 40, 50 , 100, 200, 250, 400, 600, 800, 1000 ]
sparsity = 0.1


possible_permutations = []
for i in range(len(N)):
  active_bits = N[i] * sparsity
  inactive_bits = N[i] - active_bits
  possible_permutations.append(math.factorial(N[i]) / (math.factorial(active_bits) * math.factorial(inactive_bits)))

plt.close("all")

plt.figure()
plt.plot(N, possible_permutations)
plt.yscale("log")
plt.ylabel("# of possible patterns", fontsize=12)
plt.xlabel("N", fontsize=12)
plt.savefig("numberOfPossiblePatterns" + ".png")