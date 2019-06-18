import torch
import torch.nn as nn
import torch.utils.data
import supervised_convnet
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys

# Batch size, channels, height, width

# train on 3 x 3

data = np.load("../ising81x81->27x27_using_w1_temp1_correlated.npy")[:, :9, :9]
print("data", data.shape)
# Create uncorrelated samples
uncorrelated_data = []
for _ in range(10000):
    sample = np.random.randint(0, 10000, (3, 3))
    horizontal, vertical = np.random.randint(0, 3, (2, 3, 3))
    uncorrelated = []
    for i in range(3):
        tile = []
        for j in range(3):
            tile.append(data[sample[i, j], 3*horizontal[i, j]:(3*horizontal[i, j] + 3), \
                    3*vertical[i, j]:(3*vertical[i, j] + 3)])
        uncorrelated.append(np.hstack(tile))
    uncorrelated_data.append(np.vstack(uncorrelated))


np.save("../ising81x81->9x9_using_w1_temp1_uncorrelated.npy", np.array(uncorrelated_data))
# print(sample, vertical, horizontal)