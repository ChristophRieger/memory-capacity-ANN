# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:12:11 2023

@author: chris
"""

import matplotlib.pyplot as plt
import sys

variance_iterations = 5

#  simple loss
# at N200 i increased best_loss - avg_loss from < 10 ** -4 to < 10**-5
# N =                  [10, 20, 30, 40, 50, 75 , 100, 200, 250, 400, 600, 800, 1000, 2000 ]
# number_of_patterns = [10, 30, 50, 70, 90, 160, 260, 520, 610, 975, 1280, 1720, 2030, 2500 ]

############### simple loss ############
sparsity = 0.1

N = []
number_of_patterns = []
mean_of_patterns = []
std = []
dataset_size = []

N.append(100)
number_of_patterns.append(283)
mean_of_patterns.append(243)
std.append(23.95)
dataset_size.append(300)

N.append(200)
number_of_patterns.append(580)
mean_of_patterns.append(550)
std.append(32)
dataset_size.append(600)

N.append(300)
number_of_patterns.append(865)
mean_of_patterns.append(852.2)
std.append(13.18)
dataset_size.append(900)

N.append(400)
number_of_patterns.append(1098)
mean_of_patterns.append(1083.6)
std.append(18.8)
dataset_size.append(1100)

N.append(500)
number_of_patterns.append(1494)
mean_of_patterns.append(1486.8)
std.append(7.93)
dataset_size.append(1500)

plt.close("all")
plt.figure(figsize=(10,15))
plt.subplot(3, 1, 1)
plt.suptitle("Results with BCE, {} iterations".format(variance_iterations), fontsize=14)
plt.plot(N, number_of_patterns, marker = 'o', ms = 4, label="maximum memorized patterns")
plt.errorbar(N, mean_of_patterns, std, linestyle='None', marker = 'x', ms = 4, label="mean memorized patterns with standard deviation")
plt.legend(loc="upper left")
# plt.title("", fontsize=14)
plt.ylabel("# of memorzized patterns", fontsize=12)
plt.xlabel("N", fontsize=12)
plt.tick_params(axis='both', labelsize=11)

plt.subplot(3, 1, 2)
plt.plot(N, [i / j for i, j in zip(number_of_patterns, N)], marker = 'o', ms = 4)
# plt.title("", fontsize=14)
plt.ylabel("# of memorized patterns / N", fontsize=12)
plt.xlabel("N", fontsize=12)
plt.tick_params(axis='both', labelsize=11)

plt.subplot(3, 1, 3)
plt.plot(N, [i / j**2 for i, j in zip(number_of_patterns, N)], marker = 'o', ms = 4)
# plt.title("", fontsize=14)
plt.ylabel("# of memorized patterns / $N^2$", fontsize=12)
plt.xlabel("N", fontsize=12)
plt.tick_params(axis='both', labelsize=11)
plt.savefig("bceResult" + ".png")

# sys.exit()

##### custom loss ###############
tolerance = 0.1
sparsity = 0.1
N = []
number_of_patterns = []
mean_of_patterns = []
std = []
dataset_size = []

N.append(100)
number_of_patterns.append(337)
mean_of_patterns.append(305)
std.append(24)
dataset_size.append(350)

#training right now
N.append(200)
number_of_patterns.append(653)
mean_of_patterns.append(0)
std.append(5)

N.append(300)
number_of_patterns.append(1077)
mean_of_patterns.append(1046)
std.append(26.56)
dataset_size.append(1100)

N.append(400)
number_of_patterns.append(1473)
mean_of_patterns.append(1333.2)
std.append(91.18)
dataset_size.append(1500)
# !!! tried ds 1600 with maaany epochs (600)... performed worse than ds 1500 with 200 epochs!!!

N.append(500)
number_of_patterns.append(1963)
mean_of_patterns.append(1923)
std.append(26.43)
dataset_size.append(2000)

plt.figure(figsize=(10,15))
plt.subplot(3, 1, 1)
plt.suptitle("Results with custom BCE, {} iterations".format(variance_iterations), fontsize=14)
plt.plot(N, number_of_patterns, marker = 'o', ms = 4, label="maximum memorized patterns")
plt.errorbar(N, mean_of_patterns, std, linestyle='None', marker = 'x', ms = 4, label="mean memorized patterns with standard deviation")
plt.legend(loc="upper left")
# plt.title("", fontsize=14)
plt.ylabel("# of memorzized patterns", fontsize=12)
plt.xlabel("N", fontsize=12)
plt.tick_params(axis='both', labelsize=11)

plt.subplot(3, 1, 2)
plt.plot(N, [i / j for i, j in zip(number_of_patterns, N)], marker = 'o', ms = 4)
# plt.title("", fontsize=14)
plt.ylabel("# of memorized patterns / N", fontsize=12)
plt.xlabel("N", fontsize=12)
plt.tick_params(axis='both', labelsize=11)

plt.subplot(3, 1, 3)
plt.plot(N, [i / j**2 for i, j in zip(number_of_patterns, N)], marker = 'o', ms = 4)
# plt.title("", fontsize=14)
plt.ylabel("# of memorized patterns / $N^2$", fontsize=12)
plt.xlabel("N", fontsize=12)
plt.tick_params(axis='both', labelsize=11)
plt.savefig("customBceResult" + ".png")
