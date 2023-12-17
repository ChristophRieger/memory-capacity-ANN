# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:12:11 2023

@author: chris
"""

import matplotlib.pyplot as plt

# TODO: increase resolution for small N

sparsity = 0.1

# at N200 i increased best_loss - avg_loss from < 10 ** -4 to < 10**-5
N =                  [10, 20, 30, 40, 50, 75 , 100, 200, 250, 400, 600, 800, 1000, 2000 ]
# This number is taken from the biggest dataset_size where 100% of patterns are 
# memorized
# !!! 2500 is only 99.5% accurate, but 2400 also had 99.5%
number_of_patterns = [10, 30, 50, 70, 90, 160, 260, 520, 610, 975, 1280, 1720, 2030, 2500 ]

plt.close("all")
plt.figure()  
plt.plot(N, number_of_patterns, marker = 'o', ms = 4)
plt.title("", fontsize=14)
plt.ylabel("# of memorzized patterns", fontsize=12)
plt.xlabel("N", fontsize=12)
plt.tick_params(axis='both', labelsize=11)
plt.savefig("memoryPlot" + ".png")

plt.figure()  
plt.plot(N, [i / j for i, j in zip(number_of_patterns, N)], marker = 'o', ms = 4)
plt.title("", fontsize=14)
plt.ylabel("# of memorized patterns / N", fontsize=12)
plt.xlabel("N", fontsize=12)
plt.tick_params(axis='both', labelsize=11)
plt.savefig("slope" + ".png")

plt.figure()  
plt.plot(N, [i / j**2 for i, j in zip(number_of_patterns, N)], marker = 'o', ms = 4)
plt.title("", fontsize=14)
plt.ylabel("# of memorized patterns / $N^2$", fontsize=12)
plt.xlabel("N", fontsize=12)
plt.tick_params(axis='both', labelsize=11)
plt.savefig("mByN2" + ".png")



#  custom loss!!!!!!!!
# at N200 i increased best_loss - avg_loss from < 10 ** -4 to < 10**-5
N =                  [10, 20, 30, 40, 50, 75 , 100, 200, 250, 400, 600, 800, 1000, 2000 ]
# This number is taken from the biggest dataset_size where 100% of patterns are 
# memorized
# !!! 2500 is only 99.5% accurate, but 2400 also had 99.5%
number_of_patterns = [??, 29 (mit 40) ]

plt.close("all")
plt.figure()  
plt.plot(N, number_of_patterns, marker = 'o', ms = 4)
plt.title("", fontsize=14)
plt.ylabel("# of memorzized patterns", fontsize=12)
plt.xlabel("N", fontsize=12)
plt.tick_params(axis='both', labelsize=11)
plt.savefig("memoryPlot" + ".png")

plt.figure()  
plt.plot(N, [i / j for i, j in zip(number_of_patterns, N)], marker = 'o', ms = 4)
plt.title("", fontsize=14)
plt.ylabel("# of memorized patterns / N", fontsize=12)
plt.xlabel("N", fontsize=12)
plt.tick_params(axis='both', labelsize=11)
plt.savefig("slope" + ".png")

plt.figure()  
plt.plot(N, [i / j**2 for i, j in zip(number_of_patterns, N)], marker = 'o', ms = 4)
plt.title("", fontsize=14)
plt.ylabel("# of memorized patterns / $N^2$", fontsize=12)
plt.xlabel("N", fontsize=12)
plt.tick_params(axis='both', labelsize=11)
plt.savefig("mByN2" + ".png")

