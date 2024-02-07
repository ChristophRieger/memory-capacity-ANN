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

N_simple = []
number_of_patterns_simple = []
mean_of_patterns_simple = []
std_simple = []
dataset_size = []

N_simple.append(100)
number_of_patterns_simple.append(283)
mean_of_patterns_simple.append(243)
std_simple.append(23.95)
dataset_size.append(300)

N_simple.append(200)
number_of_patterns_simple.append(580)
mean_of_patterns_simple.append(550)
std_simple.append(32)
dataset_size.append(600)

N_simple.append(300)
number_of_patterns_simple.append(897)
mean_of_patterns_simple.append(883.8)
std_simple.append(11.95)
dataset_size.append(900)

N_simple.append(400)
number_of_patterns_simple.append(1194)
mean_of_patterns_simple.append(1165.6)
std_simple.append(23.56)
dataset_size.append(1200)

N_simple.append(500)
number_of_patterns_simple.append(1494)
mean_of_patterns_simple.append(1486.8)
std_simple.append(7.93)
dataset_size.append(1500)

plt.close("all")
# plt.figure(figsize=(10,15))
# plt.subplot(3, 1, 1)
# plt.suptitle("Results with BCE, {} iterations".format(variance_iterations), fontsize=14)
# plt.plot(N_simple, number_of_patterns_simple, marker = 'o', ms = 4, label="maximum memorized patterns")
# plt.errorbar(N_simple, mean_of_patterns_simple, std_simple, linestyle='None', marker = 'x', ms = 4, label="mean memorized patterns with standard deviation")
# plt.legend(loc="upper left")
# # plt.title("", fontsize=14)
# plt.ylabel("# of memorzized patterns", fontsize=12)
# plt.xlabel("N", fontsize=12)
# plt.tick_params(axis='both', labelsize=11)

# plt.subplot(3, 1, 2)
# plt.plot(N_simple, [i / j for i, j in zip(number_of_patterns_simple, N_simple)], marker = 'o', ms = 4)
# # plt.title("", fontsize=14)
# plt.ylabel("# of memorized patterns / N", fontsize=12)
# plt.xlabel("N", fontsize=12)
# plt.tick_params(axis='both', labelsize=11)

# plt.subplot(3, 1, 3)
# plt.plot(N_simple, [i / j**2 for i, j in zip(number_of_patterns_simple, N_simple)], marker = 'o', ms = 4)
# # plt.title("", fontsize=14)
# plt.ylabel("# of memorized patterns / $N^2$", fontsize=12)
# plt.xlabel("N", fontsize=12)
# plt.tick_params(axis='both', labelsize=11)
# plt.savefig("bceResult" + ".png")

# sys.exit()

##### custom loss ###############
tolerance = 0.1
sparsity = 0.1
N_custom_loss = []
number_of_patterns_custom_loss = []
mean_of_patterns_custom_loss = []
std_custom_loss = []
dataset_size = []

N_custom_loss.append(100)
number_of_patterns_custom_loss.append(337)
mean_of_patterns_custom_loss.append(305)
std_custom_loss.append(24)
dataset_size.append(350)

N_custom_loss.append(200)
number_of_patterns_custom_loss.append(694)
mean_of_patterns_custom_loss.append(674.8)
std_custom_loss.append(20.7)
dataset_size.append(700)

N_custom_loss.append(300)
number_of_patterns_custom_loss.append(1077)
mean_of_patterns_custom_loss.append(1046)
std_custom_loss.append(26.56)
dataset_size.append(1100)

N_custom_loss.append(400)
number_of_patterns_custom_loss.append(1580)
mean_of_patterns_custom_loss.append(1395.2)
std_custom_loss.append(169)
dataset_size.append(1600)
# !!! mit viel mehr epochs kann man manchmal noch 100 weiter rauf gehn, aber es wird immer schwueriger
#  1500/200 auf 1600/600...

N_custom_loss.append(500)
number_of_patterns_custom_loss.append(2008)
mean_of_patterns_custom_loss.append(1904)
std_custom_loss.append(141.7)
dataset_size.append(2100)
# !!! 500 epochs... mby gehn 100 patterns mehr
#  hab jetzt 700 epochen mit 2100 ds gemacht... besser, aber 100 von 2100 nicht erkannt

# plt.figure(figsize=(10,15))
# plt.subplot(3, 1, 1)
# plt.suptitle("Results with custom BCE, {} iterations".format(variance_iterations), fontsize=14)
# plt.plot(N_custom_loss, number_of_patterns_custom_loss, marker = 'o', ms = 4, label="maximum memorized patterns")
# plt.errorbar(N_custom_loss, mean_of_patterns_custom_loss, std_custom_loss, linestyle='None', marker = 'x', ms = 4, label="mean memorized patterns with standard deviation")
# plt.legend(loc="upper left")
# # plt.title("", fontsize=14)
# plt.ylabel("# of memorzized patterns", fontsize=12)
# plt.xlabel("N", fontsize=12)
# plt.tick_params(axis='both', labelsize=11)

# plt.subplot(3, 1, 2)
# plt.plot(N_custom_loss, [i / j for i, j in zip(number_of_patterns_custom_loss, N_custom_loss)], marker = 'o', ms = 4)
# # plt.title("", fontsize=14)
# plt.ylabel("# of memorized patterns / N", fontsize=12)
# plt.xlabel("N", fontsize=12)
# plt.tick_params(axis='both', labelsize=11)

# plt.subplot(3, 1, 3)
# plt.plot(N_custom_loss, [i / j**2 for i, j in zip(number_of_patterns_custom_loss, N_custom_loss)], marker = 'o', ms = 4)
# # plt.title("", fontsize=14)
# plt.ylabel("# of memorized patterns / $N^2$", fontsize=12)
# plt.xlabel("N", fontsize=12)
# plt.tick_params(axis='both', labelsize=11)
# plt.savefig("customBceResult" + ".png")




##### recurrent layer + custom loss ###############
recurrances = 1
N_recurrent = []
number_of_patterns_recurrent = []
mean_of_patterns_recurrent = []
std_recurrent = []
dataset_size = []

N_recurrent.append(50)
number_of_patterns_recurrent.append(534)
mean_of_patterns_recurrent.append(524.8)
std_recurrent.append(6.18)
dataset_size.append(550)

N_recurrent.append(100)
number_of_patterns_recurrent.append(920)
mean_of_patterns_recurrent.append(902.2)
std_recurrent.append(12.5)
dataset_size.append(1000)

N_recurrent.append(150)
number_of_patterns_recurrent.append(1241)
mean_of_patterns_recurrent.append(1228)
std_recurrent.append(9.27)
dataset_size.append(1300)

N_recurrent.append(200)
number_of_patterns_recurrent.append(1549)
mean_of_patterns_recurrent.append(1539.33)
std_recurrent.append(7.13)
dataset_size.append(1600)

# plt.figure(figsize=(10,15))
# plt.subplot(3, 1, 1)
# plt.suptitle("Results with {} recurrance, custom BCE, {} iterations".format(recurrances, variance_iterations), fontsize=14)
# plt.plot(N_recurrent, number_of_patterns_recurrent, marker = 'o', ms = 4, label="maximum memorized patterns")
# plt.errorbar(N_recurrent, mean_of_patterns_recurrent, std_recurrent, linestyle='None', marker = 'x', ms = 4, label="mean memorized patterns with standard deviation")
# plt.legend(loc="upper left")
# # plt.title("", fontsize=14)
# plt.ylabel("# of memorzized patterns", fontsize=12)
# plt.xlabel("N", fontsize=12)
# plt.tick_params(axis='both', labelsize=11)

# plt.subplot(3, 1, 2)
# plt.plot(N_recurrent, [i / j for i, j in zip(number_of_patterns_recurrent, N_recurrent)], marker = 'o', ms = 4)
# # plt.title("", fontsize=14)
# plt.ylabel("# of memorized patterns / N", fontsize=12)
# plt.xlabel("N", fontsize=12)
# plt.tick_params(axis='both', labelsize=11)

# plt.subplot(3, 1, 3)
# plt.plot(N_recurrent, [i / j**2 for i, j in zip(number_of_patterns_recurrent, N_recurrent)], marker = 'o', ms = 4)
# # plt.title("", fontsize=14)
# plt.ylabel("# of memorized patterns / $N^2$", fontsize=12)
# plt.xlabel("N", fontsize=12)
# plt.tick_params(axis='both', labelsize=11)
# plt.savefig("customBceRecurrenceResult" + ".png")



##### recurrent layer, 3 steps through time + custom loss ###############
recurrances = 2
N_recurrent_2 = []
number_of_patterns_recurrent_2 = []
mean_of_patterns_recurrent_2 = []
std_recurrent_2 = []
dataset_size = []

N_recurrent_2.append(50)
number_of_patterns_recurrent_2.append(589)
mean_of_patterns_recurrent_2.append(577.66)
std_recurrent_2.append(13.3)
dataset_size.append(600)

# 
# N_recurrent_2.append(100)
# number_of_patterns_recurrent_2.append(1179)
# mean_of_patterns_recurrent_2.append(1177.33)
# std_recurrent_2.append(1.25)
# dataset_size.append(1200)
# 
N_recurrent_2.append(100)
number_of_patterns_recurrent_2.append(1219)
mean_of_patterns_recurrent_2.append(1199)
std_recurrent_2.append(14.51)
dataset_size.append(1300)

# N_recurrent_2.append(150)
# number_of_patterns_recurrent_2.append(1472)
# mean_of_patterns_recurrent_2.append(1463.67)
# std_recurrent_2.append(7.93)
# dataset_size.append(1500)
# 
N_recurrent_2.append(150)
number_of_patterns_recurrent_2.append(1528)
mean_of_patterns_recurrent_2.append(1503)
std_recurrent_2.append(17.72)
dataset_size.append(1600)

# N_recurrent_2.append(200)
# number_of_patterns_recurrent_2.append(1673)
# mean_of_patterns_recurrent_2.append(0)
# std_recurrent_2.append(0)
# dataset_size.append(1700)
N_recurrent_2.append(200)
number_of_patterns_recurrent_2.append(1950)
mean_of_patterns_recurrent_2.append(1929.33)
std_recurrent_2.append(14.7)
dataset_size.append(2100)


# combined plot
plt.figure(figsize=(10,15))
plt.plot(N_simple, number_of_patterns_simple, marker = 'o', ms = 4, label="simple", color='blue')
plt.errorbar(N_simple, mean_of_patterns_simple, std_simple, linestyle='None', marker = 'x', ms = 4, color='blue')
plt.plot(N_custom_loss, number_of_patterns_custom_loss, marker = 'o', ms = 4, label="simple + custom loss", color='green')
plt.errorbar(N_custom_loss, mean_of_patterns_custom_loss, std_custom_loss, linestyle='None', marker = 'x', ms = 4, color='green')
plt.plot(N_recurrent, number_of_patterns_recurrent, marker = 'o', ms = 4, label="recurrent layer (2 steps) + custom loss", color='red')
plt.errorbar(N_recurrent, mean_of_patterns_recurrent, std_recurrent, linestyle='None', marker = 'x', ms = 4, color='red')
plt.plot(N_recurrent_2, number_of_patterns_recurrent_2, marker = 'o', ms = 4, label="recurrent layer (3 steps) + custom loss", color='black')
plt.errorbar(N_recurrent_2, mean_of_patterns_recurrent_2, std_recurrent_2, linestyle='None', marker = 'x', ms = 4, color='black')
plt.title("Memorized patterns", fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.ylabel("# of memorzized patterns", fontsize=13)
plt.xlabel("N", fontsize=13)
plt.tick_params(axis='both', labelsize=12)
plt.savefig("combinedMemorizedPatterns" + ".png")

plt.figure(figsize=(10,15))
plt.plot(N_simple, [i / j for i, j in zip(number_of_patterns_simple, N_simple)], marker = 'o', ms = 4, label='simple', color='blue')
plt.plot(N_custom_loss, [i / j for i, j in zip(number_of_patterns_custom_loss, N_custom_loss)], marker = 'o', ms = 4, label='simple + custom loss', color='green')
plt.plot(N_recurrent, [i / j for i, j in zip(number_of_patterns_recurrent, N_recurrent)], marker = 'o', ms = 4, label='recurrent layer (2 steps) + custom loss', color='red')
plt.plot(N_recurrent_2, [i / j for i, j in zip(number_of_patterns_recurrent_2, N_recurrent_2)], marker = 'o', ms = 4, label='recurrent layer (3 steps) + custom loss', color='black')
plt.title("Memorized patterns per input neuron", fontsize=14)
plt.legend(loc="upper right", fontsize=12)
plt.ylabel("# of memorized patterns / N", fontsize=13)
plt.xlabel("N", fontsize=13)
plt.tick_params(axis='both', labelsize=12)
plt.savefig("combindedPatternPerN" + ".png")

plt.figure(figsize=(10,15))
plt.plot(N_simple, [i / j**2 for i, j in zip(number_of_patterns_simple, N_simple)], marker = 'o', ms = 4, label='simple', color='blue')
plt.plot(N_custom_loss, [i / j**2 for i, j in zip(number_of_patterns_custom_loss, N_custom_loss)], marker = 'o', ms = 4, label='simple + custom loss', color='green')
plt.plot(N_recurrent, [i / j**2 for i, j in zip(number_of_patterns_recurrent, N_recurrent)], marker = 'o', ms = 4, label='recurrent layer (2 steps) + custom loss', color='red')
plt.plot(N_recurrent_2, [i / j**2 for i, j in zip(number_of_patterns_recurrent_2, N_recurrent_2)], marker = 'o', ms = 4, label='recurrent layer (3 steps) + custom loss', color='black')
plt.title("Memorized patterns per input neuron squared", fontsize=14)
plt.legend(loc="upper right", fontsize=12)
plt.ylabel("# of memorized patterns / $N^2$", fontsize=13)
plt.xlabel("N", fontsize=13)
plt.tick_params(axis='both', labelsize=12)
plt.savefig("combindedPatternPerN2" + ".png")