# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:15:09 2023

@author: chris
"""

import math
import matplotlib.pyplot as plt
import numpy as np

N =                  [10, 20, 30, 40, 50, 75 , 100, 200, 250, 400, 600, 800, 1000, 2000 ]
number_of_patterns = [10, 30, 50, 70, 90, 160, 260, 520, 610, 975, 1280, 1720, 2030, 4000 ]
sparsity = 0.1


needed_connections = []
for i in range(len(N)):
  needed_connections.append(N[i] * sparsity * number_of_patterns[i])

plt.close("all")

plt.figure()
plt.plot(N, needed_connections)
# plt.yscale("log")
plt.ylabel("# of connections needed to memorize patterns", fontsize=12)
plt.xlabel("N", fontsize=12)
plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))

plt.savefig("numberOfUsedConnections" + ".png")