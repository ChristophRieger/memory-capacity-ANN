# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:12:11 2023

@author: chris
"""

import matplotlib.pyplot as plt

# TODO: increase resolution for small N

sparsity = 0.1

# at N200 i increased best_loss - avg_loss from < 10 ** -4 to < 10**-5
N =                  [10, 20, 30, 40, 50, 75 , 100, 200, 250, 400 ]
# This number is taken from the biggest dataset_size where 100% of patterns are 
# memorized
number_of_patterns = [10, 30, 50, 70, 90, 160, 260, 520, 610, 975 ]

plt.close("all")
plt.figure()  
plt.plot(N, number_of_patterns)
plt.title("Number of correctly memorzized patterns", fontsize=14)
plt.ylabel("patterns", fontsize=12)
plt.xlabel("N", fontsize=12)
plt.tick_params(axis='both', labelsize=11)
plt.savefig("memoryPlot" + ".svg")  
plt.savefig("memoryPlot" + ".png")