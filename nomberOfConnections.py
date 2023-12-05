# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:50:05 2023

@author: chris
"""

import math
import matplotlib.pyplot as plt
import numpy as np

N =                  [10, 20, 30, 40, 50, 75 , 100, 200, 250, 400, 600, 800, 1000, 2000 ]
# N = list(range(101))
# N = N[1:101]
sparsity = 0.1


number_of_connections = []
for i in range(len(N)):
  number_of_connections.append(N[i]**2)

plt.close("all")

plt.figure()
plt.plot(N, number_of_connections)
# plt.yscale("log")
plt.ylabel("# of connections in the networks", fontsize=12)
plt.xlabel("N", fontsize=12)
plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
plt.savefig("numberOfConnections" + ".png")